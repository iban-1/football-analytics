"""Drawing: annotated frames (boxes, IDs, trails) and the top-down radar view."""
from __future__ import annotations

from collections import defaultdict, deque

import cv2
import numpy as np
import pandas as pd

from src.video.homography import PITCH_L, PITCH_W, apply_h, pitch_lines

TEAM_BGR = {"light": (255, 200, 80), "dark": (60, 60, 200), "other": (60, 200, 230),
            "ball": (255, 255, 255)}
TRAIL_LEN = 30


def draw_detections(frame: np.ndarray, rows: pd.DataFrame, trails: dict[int, deque],
                    show_off_grass: bool = False) -> np.ndarray:
    """Boxes, track IDs and (image-space) foot trails for one frame."""
    out = frame.copy()
    for r in rows.itertuples():
        if r.cls == 0 and not r.on_grass and not show_off_grass:
            continue
        color = TEAM_BGR.get(r.team, (200, 200, 200))
        x1, y1, x2, y2 = map(int, (r.x1, r.y1, r.x2, r.y2))
        if r.cls == 32:
            cv2.circle(out, ((x1 + x2) // 2, (y1 + y2) // 2), 6, color, 2)
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if r.track_id >= 0:
            cv2.putText(out, str(r.track_id), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2, cv2.LINE_AA)
            trails[r.track_id].append(((x1 + x2) // 2, y2))
    for tid, pts in trails.items():
        if len(pts) > 1:
            cv2.polylines(out, [np.array(pts, np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def new_trails() -> dict[int, deque]:
    return defaultdict(lambda: deque(maxlen=TRAIL_LEN))


def project_trails(history: dict[int, deque], H_pitch_from_img: np.ndarray) -> dict[int, np.ndarray]:
    """Re-project each track's PITCH-space history into the current image so trails stay
    glued to the grass while the camera pans (image-space trails would slide)."""
    Hinv = np.linalg.inv(H_pitch_from_img)
    return {tid: apply_h(Hinv, np.array(pts)) for tid, pts in history.items() if len(pts) > 1}


def radar_canvas(scale: int = 8, margin: int = 20) -> np.ndarray:
    """Blank top-down pitch (105 x 68 m) with markings. Goal shown is at the LEFT (X = 0)."""
    w, h = int(PITCH_L * scale) + 2 * margin, int(PITCH_W * scale) + 2 * margin
    img = np.full((h, w, 3), (60, 130, 60), np.uint8)

    def to_px(p):  # pitch (X, Y) -> pixel; Y up on the map so flip the image axis
        return int(margin + p[0] * scale), int(margin + (PITCH_W - p[1]) * scale)

    for line in pitch_lines():
        cv2.polylines(img, [np.array([to_px(p) for p in line], np.int32)], False, (255, 255, 255), 2)
    cv2.line(img, to_px((PITCH_L / 2, 0)), to_px((PITCH_L / 2, PITCH_W)), (255, 255, 255), 2)
    cv2.circle(img, to_px((PITCH_L / 2, PITCH_W / 2)), int(9.15 * scale), (255, 255, 255), 2)
    return img


def draw_radar(pos: pd.DataFrame, scale: int = 8, margin: int = 20) -> np.ndarray:
    """Radar frame from rows with columns X, Y (pitch metres), team, track_id, cls."""
    img = radar_canvas(scale, margin)
    for r in pos.itertuples():
        if not (-5 <= r.X <= PITCH_L + 5 and -5 <= r.Y <= PITCH_W + 5):
            continue
        p = (int(margin + r.X * scale), int(margin + (PITCH_W - r.Y) * scale))
        if r.cls == 32:
            cv2.circle(img, p, 5, (255, 255, 255), -1)
            continue
        cv2.circle(img, p, 9, TEAM_BGR.get(r.team, (200, 200, 200)), -1)
        cv2.circle(img, p, 9, (0, 0, 0), 1)
        if r.track_id >= 0:
            cv2.putText(img, str(r.track_id), (p[0] - 8, p[1] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img
