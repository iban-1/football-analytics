"""Phase 7 tests on synthetic data with known answers (no YOLO weights or video needed)."""
import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")

from src.video.detect_track import feet_on_grass, shirt_colour  # noqa: E402
from src.video.homography import (alignment_score, apply_h, chain, feet_points,  # noqa: E402
                                  fit_homography, pitch_lines)
from src.video.mapping import (pitch_from_image_maps, reference_to_frame_maps,  # noqa: E402
                               track_motion_table, valid_window)
from src.video.quality import segment_quality  # noqa: E402
from src.video.segments import detect_cuts, grass_mask, segments_from_cuts  # noqa: E402
from src.video.teams import assign_teams  # noqa: E402


def _known_h():
    """A plausible pitch(metres) -> pixel homography (perspective, like a side camera)."""
    src = np.float32([[0, 0], [30, 0], [30, 40], [0, 40]])
    dst = np.float32([[100, 600], [900, 560], [1000, 250], [150, 300]])
    return cv2.getPerspectiveTransform(src, dst)


def test_fit_homography_recovers_known_mapping():
    H_pitch_to_img = _known_h()
    pitch = np.array([[2, 3], [25, 4], [28, 35], [3, 38], [15, 20], [10, 30]], float)
    pixels = apply_h(H_pitch_to_img, pitch)
    H, resid = fit_homography(pixels, pitch)
    assert resid.max() < 1e-4   # metres; the reference homography is float32
    assert np.allclose(apply_h(H, apply_h(H_pitch_to_img, np.array([[20.0, 10.0]]))), [[20, 10]],
                       atol=1e-4)


def test_static_point_keeps_its_pitch_position_while_camera_pans():
    """Camera pans 7 px/frame; a fixed pitch spot moves in the image but its mapped
    pitch position must not change when we chain the camera homographies."""
    H_pitch_to_img = _known_h()
    H_ref = np.linalg.inv(H_pitch_to_img)           # image(ref) -> pitch
    shift = np.array([[1, 0, -7.0], [0, 1, 0], [0, 0, 1]])   # frame k -> k+1: image slides left
    seg_start, ref, n = 100, 103, 8
    Hs = [shift] * n
    maps = reference_to_frame_maps(Hs, seg_start, ref)
    Hpi = pitch_from_image_maps(H_ref, maps)
    spot_pitch = np.array([[12.0, 9.0]])
    spot_ref_px = apply_h(H_pitch_to_img, spot_pitch)
    for t in range(seg_start, seg_start + n + 1):
        px_t = apply_h(maps[t], spot_ref_px)
        assert np.allclose(apply_h(Hpi[t], px_t), spot_pitch, atol=1e-6)
        assert px_t[0, 0] == pytest.approx(spot_ref_px[0, 0] - 7 * (t - ref))


def test_chain_composes_in_order():
    a = np.array([[1, 0, 2.0], [0, 1, 0], [0, 0, 1]])
    b = np.array([[1, 0, 0], [0, 1, 3.0], [0, 0, 1]])
    out = chain([a, b])
    assert np.allclose(out[0], np.eye(3)) and np.allclose(out[2][:2, 2], [2, 3])


def test_feet_points_bottom_centre():
    assert np.allclose(feet_points(np.array([[10, 20, 30, 80]])), [[20, 80]])


def test_valid_window_stops_at_first_failure():
    scores = {f: (5.0 if 100 <= f <= 110 else 1.0) for f in range(90, 125)}
    assert valid_window(scores, ref=105, threshold=3.0) == (100, 110)
    nan_scores = {**scores, 107: float("nan")}   # a single noisy frame must not end the window
    assert valid_window(nan_scores, ref=105, threshold=3.0) == (100, 110)


def test_alignment_score_high_for_true_lines_low_for_wrong_ones():
    H_pitch_to_img = _known_h()
    img = np.zeros((720, 1280), np.uint8)
    for line in pitch_lines()[:3]:                       # draw box + arc as thin white lines
        pts = apply_h(H_pitch_to_img @ np.diag([1, 1, 1]), line)
        cv2.polylines(img, [pts.astype(np.int32)], False, 255, 2)
    H_img_to_pitch = np.linalg.inv(H_pitch_to_img)
    good = alignment_score(img, H_img_to_pitch)
    wrong = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ np.linalg.inv(
        np.array([[1, 0, 40.0], [0, 1, 60.0], [0, 0, 1]]) @ H_pitch_to_img)
    bad = alignment_score(img, wrong)
    assert (np.isnan(good) or good > 3.0) and not (bad > 3.0 if not np.isnan(bad) else False)


def test_track_motion_constant_speed():
    fps = 25.0
    frames = np.arange(100, 200)
    pos = pd.DataFrame({"frame": frames, "track_id": 7, "cls": 0, "team": "light",
                        "X": 10 + 5.0 * (frames - 100) / fps, "Y": 20.0})
    m = track_motion_table(pos, fps).iloc[0]
    assert m["mean_speed_ms"] == pytest.approx(5.0, abs=0.05)
    assert m["distance_m"] == pytest.approx(5.0 * 99 / fps * (100 / 99), abs=0.5)
    assert not m["implausible_speed"]


def test_track_motion_flags_teleport_and_skips_short_or_gappy_tracks():
    fps = 25.0
    frames = np.arange(0, 100)
    x = np.where(frames < 50, 10.0, 60.0)                 # 50 m jump = mapping error
    pos = pd.DataFrame({"frame": frames, "track_id": 1, "cls": 0, "team": "dark", "X": x, "Y": 20.0})
    short = pos.iloc[:20].assign(track_id=2)
    gappy = pos[(pos["frame"] < 40) | (pos["frame"] > 60)].assign(track_id=3)
    out = track_motion_table(pd.concat([pos, short, gappy]), fps)
    assert out["track_id"].tolist() == [1] and bool(out.iloc[0]["implausible_speed"])


def test_assign_teams_two_kits_and_outlier():
    rng = np.random.default_rng(0)
    rows = []
    for tid in range(20):
        light = tid < 10
        lab = (200, 120, 110) if light else (60, 130, 100)
        for f in range(10):
            rows.append((0, f, tid, 0, 0.9, 0, 0, 10, 20, True,
                         lab[0] + rng.normal(0, 3), lab[1] + rng.normal(0, 2), lab[2] + rng.normal(0, 2)))
    for f in range(10):                                   # a referee in a very different colour
        rows.append((0, f, 99, 0, 0.9, 0, 0, 10, 20, True, 90 + rng.normal(0, 2), 190, 190))
    det = pd.DataFrame(rows, columns=["segment", "frame", "track_id", "cls", "conf", "x1", "y1",
                                      "x2", "y2", "on_grass", "L", "a", "b"])
    out = assign_teams(det, seed=0)
    team = out.drop_duplicates("track_id").set_index("track_id")["team"]
    assert (team.loc[:9] == "light").all() and (team.loc[10:19] == "dark").all()
    assert team.loc[99] == "other"


def test_feet_on_grass_and_shirt_colour():
    img = np.zeros((200, 200, 3), np.uint8)
    img[:] = (40, 160, 40)                                # BGR green pitch
    img[40:100, 80:120] = (255, 255, 255)                 # a white-shirted player
    green = grass_mask(img)
    on = np.array([80, 40, 120, 100], float)
    crowd = np.array([80, 5, 120, 35], float)
    assert feet_on_grass(green, on)
    img2 = np.full((200, 200, 3), (60, 60, 200), np.uint8)   # red stand, no grass
    assert not feet_on_grass(grass_mask(img2), crowd)
    L, a, b = shirt_colour(img, green, on)
    assert L > 230                                        # white torso, grass pixels ignored


def test_segment_quality_proxies():
    rows = []
    for f in range(10):
        for tid in (1, 2):
            rows.append((0, f, tid, 0, 0.9, 0, 0, 1, 1, True))
    rows.append((0, 3, -1, 32, 0.5, 0, 0, 1, 1, True))
    rows.append((0, 3, 5, 0, 0.5, 0, 0, 1, 1, False))     # steward off the grass
    det = pd.DataFrame(rows, columns=["segment", "frame", "track_id", "cls", "conf", "x1", "y1",
                                      "x2", "y2", "on_grass"])
    q = segment_quality(det, 10)
    assert q["unique_track_ids"] == 2 and q["median_players_per_frame"] == 2.0
    assert q["id_to_player_ratio"] == 1.0 and q["ball_frames_share"] == 0.1
    assert q["persons_removed_not_on_grass"] == 1


def test_cut_between_similar_colour_scenes_is_caught_by_pixel_spike(tmp_path):
    """Horizontal vs vertical stripes: IDENTICAL colour histograms, so only the pixel-change
    signal can see this cut (like a wide pitch shot cutting to a tight pitch shot)."""
    path = tmp_path / "similar.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (160, 90))
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:90, 0:160]
    horizontal = np.where((yy // 6) % 2 == 0, 60, 180).astype(np.uint8)
    vertical = np.where((xx // 6) % 2 == 0, 60, 180).astype(np.uint8)
    for base, n in ((horizontal, 40), (vertical, 40)):
        for _ in range(n):
            g = np.clip(base + rng.integers(0, 3, base.shape), 0, 255).astype(np.uint8)
            w.write(cv2.merge([g, g, g]))
    w.release()
    assert detect_cuts(path) == [40]


def test_segments_and_cut_detection(tmp_path):
    path = tmp_path / "cuts.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (160, 90))
    rng = np.random.default_rng(0)
    for color, n in (((0, 160, 0), 40), ((200, 40, 40), 40), ((30, 30, 220), 40)):
        for _ in range(n):
            frame = np.full((90, 160, 3), color, np.uint8)
            frame = np.clip(frame + rng.integers(0, 6, frame.shape), 0, 255).astype(np.uint8)
            w.write(frame)
    w.release()
    cuts = detect_cuts(path)
    assert cuts == [40, 80]
    assert segments_from_cuts(cuts, 120) == [(0, 40), (40, 80), (80, 120)]
    assert segments_from_cuts([10], 120, min_len=25) == [(10, 120)]
