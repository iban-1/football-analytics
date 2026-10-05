"""Split a broadcast clip into camera shots.

A highlights edit cuts between many cameras, and a tracker's IDs and a pitch homography
are only valid within ONE continuous shot, so everything downstream works shot by shot.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def detect_cuts(video: Path, threshold: float = 0.35, spike_min: float = 10.0,
                spike_ratio: float = 3.5) -> list[int]:
    """Frame indices where a new shot starts. Two signals are combined (union):

    1. COLOUR: each frame is shrunk to 160 x 90, converted to HSV, and its hue/saturation
       histogram compared with the previous frame's (Bhattacharyya distance, 0 = identical,
       1 = disjoint). A cut between very different scenes pushes it above `threshold`.
    2. PIXELS: mean absolute grey-level change between consecutive frames. A hard cut makes
       it jump to at least `spike_min` AND to `spike_ratio` times the median of the 12
       neighbouring frames. This catches cuts between two similar-looking scenes (wide
       pitch shot -> tight pitch shot), whose colour histograms barely differ; the colour
       test alone missed exactly such a cut in the demo clip.
    Fades and very fast pans can still be missed or split.
    """
    cap = cv2.VideoCapture(str(video))
    prev_hist, prev_grey = None, None
    hist_cuts, diffs = set(), []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (160, 90))
        hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 16],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        grey = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
        if prev_hist is not None:
            if cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > threshold:
                hist_cuts.add(i)
            diffs.append(float(np.abs(grey - prev_grey).mean()))
        else:
            diffs.append(0.0)
        prev_hist, prev_grey, i = hist, grey, i + 1
    cap.release()
    d = np.array(diffs)
    spike_cuts = set()
    for t in range(1, len(d)):
        neighbours = np.r_[d[max(1, t - 6):t], d[t + 1:t + 7]]
        if len(neighbours) and d[t] > spike_min and d[t] > spike_ratio * max(np.median(neighbours), 1.0):
            spike_cuts.add(t)
    return sorted(hist_cuts | spike_cuts)


def segments_from_cuts(cuts: list[int], n_frames: int, min_len: int = 25,
                       ) -> list[tuple[int, int]]:
    """[start, end) frame ranges between cuts, keeping only shots of at least `min_len` frames."""
    bounds = [0] + list(cuts) + [n_frames]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= min_len]


def video_info(video: Path) -> dict:
    cap = cv2.VideoCapture(str(video))
    info = {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


def grass_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Boolean mask of green pitch pixels (hue 35-85, reasonably saturated and bright)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return (hsv[..., 0] >= 35) & (hsv[..., 0] <= 85) & (hsv[..., 1] >= 50) & (hsv[..., 2] >= 40)
