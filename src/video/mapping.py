"""Glue between camera-motion homographies and pitch positions (pure functions, testable)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.video.homography import apply_h, feet_points


def reference_to_frame_maps(Hs: list[np.ndarray], seg_start: int, ref: int,
                            ) -> dict[int, np.ndarray]:
    """Map from reference-frame pixels to the pixels of every frame of the shot.

    `Hs[k]` maps frame (seg_start + k) -> (seg_start + k + 1). Going forward we multiply
    these; going backward we multiply their inverses. The reference maps to itself.
    """
    maps: dict[int, np.ndarray] = {ref: np.eye(3)}
    last = seg_start + len(Hs)  # last frame index in the shot
    for t in range(ref + 1, last + 1):
        maps[t] = Hs[t - 1 - seg_start] @ maps[t - 1]
    for t in range(ref - 1, seg_start - 1, -1):
        maps[t] = np.linalg.inv(Hs[t - seg_start]) @ maps[t + 1]
    return maps


def pitch_from_image_maps(H_ref: np.ndarray, ref_maps: dict[int, np.ndarray],
                          ) -> dict[int, np.ndarray]:
    """Image(t) -> pitch homography per frame: H_ref composed with the inverse camera map."""
    return {t: H_ref @ np.linalg.inv(M) for t, M in ref_maps.items()}


def valid_window(scores: dict[int, float], ref: int, threshold: float, smooth: int = 5,
                 ) -> tuple[int, int]:
    """Contiguous frame range around `ref` whose (median-smoothed) alignment score >= threshold.

    Returns (first, last) inclusive. NaN scores count as failing. Smoothing over `smooth`
    frames stops a single noisy frame from ending the window early.
    """
    frames = sorted(scores)
    s = pd.Series([scores[f] for f in frames], index=frames).rolling(
        smooth, center=True, min_periods=1).median()
    ok = (s >= threshold).to_dict()
    first = last = ref
    while ok.get(first - 1, False):
        first -= 1
    while ok.get(last + 1, False):
        last += 1
    return first, last


def track_motion_table(pos: pd.DataFrame, fps: float, min_frames: int = 50,
                       max_gap: int = 2, max_plausible_ms: float = 12.0) -> pd.DataFrame:
    """Distance and speed per track from mapped pitch positions (UNVALIDATED estimates).

    Positions are smoothed with a Savitzky-Golay filter (window 11 frames, quadratic)
    before differentiating, because box-corner jitter and homography noise would
    otherwise be amplified into absurd speeds. Gaps of up to `max_gap` frames are
    bridged linearly; a track with a longer gap is skipped. Tracks whose peak speed
    still exceeds `max_plausible_ms` are flagged, not silently dropped.
    """
    from scipy.signal import savgol_filter
    rows = []
    for tid, g in pos[(pos["cls"] == 0) & (pos["track_id"] >= 0)].groupby("track_id"):
        g = g.sort_values("frame").drop_duplicates("frame")
        if len(g) < min_frames:
            continue
        frames = g["frame"].to_numpy()
        if np.diff(frames).max() > max_gap + 1:
            continue
        full = np.arange(frames[0], frames[-1] + 1)
        x = np.interp(full, frames, g["X"].to_numpy())
        y = np.interp(full, frames, g["Y"].to_numpy())
        dt = 1.0 / fps
        vx = savgol_filter(x, 11, 2, deriv=1, delta=dt)
        vy = savgol_filter(y, 11, 2, deriv=1, delta=dt)
        speed = np.hypot(vx, vy)
        rows.append({"track_id": int(tid), "team": g["team"].mode().iloc[0],
                     "frames": len(full), "seconds": round(len(full) * dt, 1),
                     "distance_m": round(float(speed.sum() * dt), 1),
                     "mean_speed_ms": round(float(speed.mean()), 2),
                     "max_speed_ms": round(float(speed.max()), 2),
                     "implausible_speed": bool(speed.max() > max_plausible_ms)})
    return pd.DataFrame(rows)


def players_to_pitch(det: pd.DataFrame, Hpi: dict[int, np.ndarray]) -> pd.DataFrame:
    """Add pitch coordinates X, Y (metres) from each box's feet using that frame's homography.

    Persons use the bottom-centre of the box (ground contact). The ball box centre is used
    for the ball, but note a ball in the air is NOT on the ground plane, so its pitch
    position is only a projection along the camera ray.
    """
    parts = []
    for frame, g in det.groupby("frame"):
        if frame not in Hpi:
            continue
        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy()
        pts = feet_points(boxes)
        ball = (g["cls"] == 32).to_numpy()
        pts[ball] = np.c_[(boxes[ball, 0] + boxes[ball, 2]) / 2, (boxes[ball, 1] + boxes[ball, 3]) / 2]
        xy = apply_h(Hpi[frame], pts)
        parts.append(g.assign(X=xy[:, 0], Y=xy[:, 1]))
    return pd.concat(parts) if parts else det.assign(X=np.nan, Y=np.nan).iloc[0:0]
