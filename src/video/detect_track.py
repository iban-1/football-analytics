"""Player/ball detection and tracking with a pretrained YOLO model (Ultralytics).

Design in plain words
---------------------
* YOLO (COCO-pretrained, nothing trained by us) finds `person` and `sports ball` boxes.
* A tracker (ByteTrack or BoT-SORT, both shipped with Ultralytics) links boxes across
  frames into IDs. It is reset at every cut because IDs cannot survive a camera change.
* A person counts as a PLAYER candidate only if the strip just below the feet is mostly
  grass. This removes stewards, photographers and spectators at the pitch edge.
* While tracking we also record each box's median shirt colour (grass pixels excluded)
  so players can later be split into two teams.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from src.video.homography import frame_to_frame_h, mask_for_features
from src.video.segments import grass_mask

PERSON, BALL = 0, 32          # COCO class ids
COLUMNS = ["segment", "frame", "track_id", "cls", "conf", "x1", "y1", "x2", "y2",
           "on_grass", "L", "a", "b"]


def feet_on_grass(green: np.ndarray, box: np.ndarray, min_fraction: float = 0.35) -> bool:
    """True if the 12-pixel strip under the box (or its bottom edge) is mostly green."""
    h, w = green.shape
    x1, y1, x2, y2 = box.astype(int)
    x1, x2 = max(0, x1), min(w, x2)
    top, bottom = (y2, min(h, y2 + 12)) if y2 + 4 < h else (max(0, y2 - 12), h)
    strip = green[top:bottom, x1:x2]
    return bool(strip.size and strip.mean() >= min_fraction)


def shirt_colour(frame: np.ndarray, green: np.ndarray, box: np.ndarray,
                 min_pixels: int = 40) -> tuple[float, float, float]:
    """Median Lab colour of the torso (central 50% width, 15-55% height), grass removed."""
    x1, y1, x2, y2 = box.astype(int)
    w, h = x2 - x1, y2 - y1
    tx1, tx2 = x1 + int(0.25 * w), x2 - int(0.25 * w)
    ty1, ty2 = y1 + int(0.15 * h), y1 + int(0.55 * h)
    crop = frame[max(0, ty1):ty2, max(0, tx1):tx2]
    gmask = green[max(0, ty1):ty2, max(0, tx1):tx2]
    if crop.size == 0:
        return (np.nan, np.nan, np.nan)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)[~gmask.reshape(-1)]
    if len(lab) < min_pixels:
        return (np.nan, np.nan, np.nan)
    med = np.median(lab, axis=0)
    return float(med[0]), float(med[1]), float(med[2])


def track_segment(model, cap: cv2.VideoCapture, start: int, end: int, segment: int,
                  tracker: str = "bytetrack.yaml", conf: float = 0.15, imgsz: int = 1280,
                  static_regions: list[tuple[int, int, int, int]] | None = None,
                  want_camera: bool = False) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Run detection + tracking on frames [start, end) of one shot.

    Returns (detections table, list of frame-to-frame camera homographies). The
    homography list has one entry per consecutive frame pair when `want_camera` is set
    (players are masked out of the background-motion estimate).
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    rows, Hs = [], []
    prev_gray, prev_mask = None, None
    for f in range(start, end):
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(frame, persist=(f > start), tracker=tracker, classes=[PERSON, BALL],
                          conf=conf, imgsz=imgsz, verbose=False)[0]
        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.zeros((0, 4))
        cls = boxes.cls.cpu().numpy().astype(int) if len(boxes) else np.zeros(0, int)
        scores = boxes.conf.cpu().numpy() if len(boxes) else np.zeros(0)
        ids = (boxes.id.cpu().numpy().astype(int)
               if len(boxes) and boxes.id is not None else -np.ones(len(cls), int))
        green = grass_mask(frame)
        for box, c, s, i in zip(xyxy, cls, scores, ids):
            if c == PERSON:
                lab = shirt_colour(frame, green, box)
                grass = feet_on_grass(green, box)
            else:
                lab, grass = (np.nan, np.nan, np.nan), True
            rows.append((segment, f, i, c, s, *box, grass, *lab))
        if want_camera:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                H, _ = frame_to_frame_h(prev_gray, gray, prev_mask)
                Hs.append(H)
            persons = xyxy[cls == PERSON] if len(xyxy) else None
            prev_gray = gray
            prev_mask = mask_for_features(gray.shape, persons, static_regions or [])
    return pd.DataFrame(rows, columns=COLUMNS), Hs


def is_wide_shot(model, frame: np.ndarray, imgsz: int = 1280, max_height_frac: float = 0.28,
                 min_players: int = 5) -> tuple[bool, dict]:
    """Heuristic: a gameplay camera shows many SMALL players standing on grass.

    Detect persons in one frame, keep those on grass, and call the shot "wide" if there
    are at least `min_players` and their median box height is under 28% of the frame.
    Close-ups of faces, the referee or the trophy fail this test.
    """
    res = model(frame, classes=[PERSON], conf=0.25, imgsz=imgsz, verbose=False)[0]
    xyxy = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
    green = grass_mask(frame)
    keep = np.array([feet_on_grass(green, b) for b in xyxy], bool) if len(xyxy) else np.zeros(0, bool)
    heights = (xyxy[keep, 3] - xyxy[keep, 1]) / frame.shape[0] if keep.any() else np.zeros(0)
    info = {"players_on_grass": int(keep.sum()),
            "median_height_frac": round(float(np.median(heights)), 3) if len(heights) else None}
    ok = bool(len(heights) >= min_players and np.median(heights) < max_height_frac)
    return ok, info
