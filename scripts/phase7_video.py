"""Phase 7 (optional): player detection, tracking and pitch mapping on a broadcast clip.

Run from the repo root:  python -m scripts.phase7_video
Needs the extra packages in requirements-video.txt and the clip named in config.yaml
(`video.path`). Outputs go to results/phase7/. Videos and any frames from the clip are
git-ignored; tables and derived plots are committed.
"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO

from src.utils.config import ROOT, load_config, resolve, set_seed
from src.video.detect_track import is_wide_shot, track_segment
from src.video.homography import alignment_score, fit_homography
from src.video.mapping import (pitch_from_image_maps, players_to_pitch, reference_to_frame_maps,
                               track_motion_table, valid_window)
from src.video.quality import segment_quality
from src.video.render import (draw_detections, draw_radar, new_trails, project_trails)
from src.video.segments import detect_cuts, segments_from_cuts, video_info
from src.video.teams import assign_teams
from src.viz.tracking_plots import tracking_pitch

TRACKERS = ("bytetrack.yaml", "botsort.yaml")


def heatmap_png(pos: pd.DataFrame, title: str, path) -> None:
    """Smoothed density of mapped player positions (pitch metres, goal at the left)."""
    fig, ax, pitch = tracking_pitch()
    stat = pitch.bin_statistic(pos["X"], pos["Y"], statistic="count", bins=(21, 14))
    stat["statistic"] = gaussian_filter(stat["statistic"].astype(float), 1.2)
    hm = pitch.heatmap(stat, ax=ax, cmap="YlOrRd", edgecolors=None, alpha=0.9)
    fig.colorbar(hm, ax=ax, shrink=0.6, label="player-frames per cell (smoothed)")
    ax.set_title(title)
    fig.savefig(path, dpi=130, bbox_inches="tight")


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    v = cfg["video"]
    set_seed(cfg["seed"])
    out = resolve(cfg["paths"]["results"]) / "phase7"
    out.mkdir(parents=True, exist_ok=True)
    path = ROOT / v["path"]
    info = video_info(path)
    fps = info["fps"]
    print("Video:", info)
    summary: dict = {"video": info}

    # ---- 1. shots ----
    cuts = detect_cuts(path, v["cut_threshold"])
    segments = segments_from_cuts(cuts, info["frames"])
    print(f"{len(cuts)} cuts -> {len(segments)} shots of >= 1 s")
    model = YOLO(v["model"])
    cap = cv2.VideoCapture(str(path))
    shot_rows = []
    for i, (a, b) in enumerate(segments):
        cap.set(cv2.CAP_PROP_POS_FRAMES, (a + b) // 2)
        ok, frame = cap.read()
        wide, det_info = is_wide_shot(model, frame, v["imgsz"]) if ok else (False, {})
        shot_rows.append({"segment": i, "start": a, "end": b, "seconds": round((b - a) / fps, 1),
                          "wide_gameplay_shot": wide, **det_info})
    shots = pd.DataFrame(shot_rows)
    shots.to_csv(out / "shots.csv", index=False)
    print(shots.to_string(index=False))
    wide_ids = shots.loc[shots["wide_gameplay_shot"], "segment"].tolist()
    summary["shots_total"], summary["shots_wide"] = len(shots), len(wide_ids)
    summary["wide_seconds"] = float(shots.loc[shots["wide_gameplay_shot"], "seconds"].sum())

    # ---- 2. mapped shot ----
    ms = v["mapped_shot"]
    mapped_seg = int(shots[(shots["start"] <= ms["reference_frame"])
                           & (shots["end"] > ms["reference_frame"])]["segment"].iloc[0])
    a_m, b_m = int(shots.loc[mapped_seg, "start"]), int(shots.loc[mapped_seg, "end"])
    assert (a_m, b_m) == (ms["start"], ms["end"]), "config mapped_shot bounds != detected shot"

    # ---- 3. compare trackers on the mapped shot (proxy metrics only) ----
    comp = []
    for trk in TRACKERS:
        d, _ = track_segment(model, cap, a_m, b_m, mapped_seg, trk, v["conf"], v["imgsz"])
        q = segment_quality(d, b_m - a_m)
        comp.append({"tracker": trk, **q})
        print(f"[{time.time() - t0:5.0f}s] tracker {trk}: {q}")
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "tracker_comparison.csv", index=False)
    chosen = comp.sort_values(["id_to_player_ratio", "tracker"]).iloc[0]["tracker"]
    summary["tracker_chosen"] = chosen
    print("Chosen tracker (lowest ID-to-player ratio, a fragmentation proxy):", chosen)

    # ---- 4. track every wide shot with the chosen tracker ----
    all_det, quality, Hs_mapped = [], [], None
    for sid in wide_ids:
        a, b = int(shots.loc[sid, "start"]), int(shots.loc[sid, "end"])
        want = sid == mapped_seg
        d, Hs = track_segment(model, cap, a, b, sid, chosen, v["conf"], v["imgsz"],
                              [tuple(r) for r in v["static_regions"]], want_camera=want)
        if want:
            Hs_mapped = Hs
        all_det.append(d)
        quality.append({"segment": sid, "start": a, "end": b, **segment_quality(d, b - a)})
        print(f"[{time.time() - t0:5.0f}s] tracked shot {sid} frames {a}-{b}")
    det = assign_teams(pd.concat(all_det, ignore_index=True), cfg["seed"])
    det.to_csv(out / "detections.csv", index=False)
    pd.DataFrame(quality).to_csv(out / "tracking_quality.csv", index=False)
    summary["team_counts_tracks"] = (det[(det["cls"] == 0) & det["on_grass"].astype(bool)
                                         & (det["track_id"] >= 0)]
                                     .drop_duplicates(["segment", "track_id"])["team"]
                                     .value_counts().to_dict())
    summary["kit_lab_centres"] = det.attrs.get("kit_lab_centres")

    # ---- 5. homography + valid window on the mapped shot ----
    lm = ms["landmarks"]
    px = np.array([m["pixel"] for m in lm], float)
    pt = np.array([m["pitch"] for m in lm], float)
    H_ref, resid = fit_homography(px, pt)
    summary["landmark_fit_residual_m"] = {"mean": round(float(resid.mean()), 3),
                                         "max": round(float(resid.max()), 3), "n": len(lm)}
    ref = ms["reference_frame"]
    maps = reference_to_frame_maps(Hs_mapped, a_m, ref)
    Hpi_all = pitch_from_image_maps(H_ref, maps)
    scores = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, a_m)
    for f in range(a_m, a_m + len(Hs_mapped) + 1):
        ok, frame = cap.read()
        if ok:
            scores[f] = alignment_score(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), Hpi_all[f])
    pd.Series(scores, name="alignment_score").to_csv(out / "alignment_scores.csv",
                                                      index_label="frame")
    first, last = valid_window(scores, ref, ms["min_alignment"])
    summary["mapped_window"] = {"first_frame": first, "last_frame": last,
                                "seconds": round((last - first + 1) / fps, 2),
                                "share_of_shot": round((last - first + 1) / (b_m - a_m), 3),
                                "min_alignment_threshold": ms["min_alignment"],
                                "alignment_at_reference": round(float(scores[ref]), 2)}
    print("Mapped window:", summary["mapped_window"])
    Hpi = {f: H for f, H in Hpi_all.items() if first <= f <= last}
    seg_det = det[(det["segment"] == mapped_seg) & (det["frame"].between(first, last))]
    pos = players_to_pitch(seg_det[seg_det["on_grass"].astype(bool) | (seg_det["cls"] == 32)], Hpi)
    pos.to_csv(out / "pitch_positions.csv", index=False)
    pl = pos[(pos["cls"] == 0) & pos["X"].between(-3, 108) & pos["Y"].between(-3, 71)]
    for team in ("light", "dark"):
        t = pl[pl["team"] == team]
        if len(t) > 20:
            heatmap_png(t, f"Mapped positions, {team} kit ({len(t)} player-frames, "
                           f"frames {first}-{last})", out / f"heatmap_{team}.png")
    motion = track_motion_table(pos, fps)
    motion.to_csv(out / "track_motion.csv", index=False)
    summary["motion"] = {"tracks": int(len(motion)),
                         "implausible_speed_tracks": int(motion["implausible_speed"].sum())
                         if len(motion) else 0}
    print(motion.to_string(index=False) if len(motion) else "no tracks long enough for motion")

    # ---- 6. videos ----
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = info["width"], info["height"]
    vw = cv2.VideoWriter(str(out / "annotated.mp4"), fourcc, fps, (w, h))
    rw = None
    by_frame = {f: g for f, g in det.groupby("frame")}
    seg_of = {}
    for sid in wide_ids:
        for f in range(int(shots.loc[sid, "start"]), int(shots.loc[sid, "end"])):
            seg_of[f] = sid
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    trails, prev_seg = new_trails(), None
    history: dict[int, list] = {}
    pos_by_frame = {f: g for f, g in pos.groupby("frame")}
    for f in range(info["frames"]):
        ok, frame = cap.read()
        if not ok:
            break
        sid = seg_of.get(f)
        if sid != prev_seg:
            trails, history = new_trails(), {}
        prev_seg = sid
        if sid is None:
            note = "close-up / non-gameplay shot: not analysed"
            vis = frame
        else:
            vis = draw_detections(frame, by_frame.get(f, det.iloc[0:0]), trails)
            note = f"shot {sid}: tracked ({chosen.split('.')[0]})"
            if f in Hpi and f in pos_by_frame:       # trails glued to the grass
                for r in pos_by_frame[f][pos_by_frame[f]["cls"] == 0].itertuples():
                    if r.track_id >= 0:
                        history.setdefault(r.track_id, []).append((r.X, r.Y))
                        history[r.track_id] = history[r.track_id][-30:]
                for pts in project_trails({k: v_ for k, v_ in history.items()}, Hpi[f]).values():
                    cv2.polylines(vis, [pts.astype(np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)
                note += "  | pitch-mapped"
        cv2.putText(vis, note, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, note, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(vis)
        if f in pos_by_frame:
            rad = draw_radar(pos_by_frame[f])
            if rw is None:
                rw = cv2.VideoWriter(str(out / "radar.mp4"), fourcc, fps, (rad.shape[1], rad.shape[0]))
            rw.write(rad)
    vw.release()
    if rw is not None:
        rw.release()
    cap.release()

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nDone in {time.time() - t0:.0f}s. Outputs in {out}")


if __name__ == "__main__":
    main()
