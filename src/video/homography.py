"""Image <-> pitch mapping for a broadcast shot (Phase 7).

Pitch frame used here (metres, plan view): X runs from the goal line of the visible
goal (X = 0) toward the halfway line (X = 52.5); Y runs from the NEAR touchline
(Y = 0, camera side) to the FAR touchline (Y = 68). The goal centre is (0, 34).
This is a proper (non-mirrored) top-down map of what the camera sees when the visible
goal is on the left of the picture.

Method
------
1. A homography H_ref maps pixels of ONE reference frame to the pitch, fitted from
   hand-picked landmark pairs (pixel <-> known pitch position of a line corner).
2. The broadcast camera pans and zooms from a fixed position. Image motion from such a
   camera is itself a homography, so we track static background points between
   consecutive frames and chain the frame-to-frame homographies back to the reference.
3. A player's feet (bottom-centre of the box) are mapped to the pitch with the
   chained homography of that frame.
Errors accumulate along the chain (drift) and players off the ground plane (jumping)
are mapped wrongly; both are measured/reported, not assumed away.
"""
from __future__ import annotations

import cv2
import numpy as np

PITCH_L, PITCH_W = 105.0, 68.0
GOAL_HALF = 3.66          # goal is 7.32 m wide
SIX_DEPTH, SIX_HALF = 5.5, 9.16     # 6-yard box: 5.5 m deep, 18.32 m wide
BOX_DEPTH, BOX_HALF = 16.5, 20.16   # penalty area: 16.5 m deep, 40.32 m wide
SPOT_X, ARC_R = 11.0, 9.15
CY = PITCH_W / 2


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to an (N, 2) array of points."""
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def fit_homography(pixels: np.ndarray, pitch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares homography pixel -> pitch from >= 4 correspondences.

    Returns (H, residuals_in_metres) where residuals are the distances between each
    landmark's known pitch position and where H sends its pixel (a fit error).
    """
    H, _ = cv2.findHomography(np.asarray(pixels, np.float64), np.asarray(pitch, np.float64), 0)
    resid = np.linalg.norm(apply_h(H, pixels) - np.asarray(pitch), axis=1)
    return H, resid


def pitch_lines() -> list[np.ndarray]:
    """Polylines (in pitch metres) of the markings near the visible goal, for overlays."""
    box = lambda d, h: np.array([[0, CY - h], [d, CY - h], [d, CY + h], [0, CY + h]], float)
    t = np.linspace(-np.arccos((BOX_DEPTH - SPOT_X) / ARC_R), np.arccos((BOX_DEPTH - SPOT_X) / ARC_R), 40)
    arc = np.c_[SPOT_X + ARC_R * np.cos(t), CY + ARC_R * np.sin(t)]
    return [box(SIX_DEPTH, SIX_HALF), box(BOX_DEPTH, BOX_HALF), arc,
            np.array([[0, CY - GOAL_HALF], [0, CY + GOAL_HALF]]),
            np.array([[0, 0], [PITCH_L, 0]]), np.array([[0, PITCH_W], [PITCH_L, PITCH_W]]),
            np.array([[0, 0], [0, PITCH_W]])]


def draw_pitch_overlay(frame: np.ndarray, H_pitch_from_img: np.ndarray,
                       color=(0, 0, 255)) -> np.ndarray:
    """Project the pitch markings into the image to check the homography by eye."""
    out = frame.copy()
    Hinv = np.linalg.inv(H_pitch_from_img)
    for line in pitch_lines():
        pts = apply_h(Hinv, line)
        if np.all(np.abs(pts) < 1e5):
            cv2.polylines(out, [pts.astype(np.int32)], False, color, 2, cv2.LINE_AA)
    return out


# ---------- camera motion between consecutive frames ----------

def mask_for_features(shape: tuple[int, int], boxes: np.ndarray | None,
                      static_regions: list[tuple[int, int, int, int]]) -> np.ndarray:
    """255 where background features may be taken; players and TV graphics are blanked.

    `static_regions` are (x1, y1, x2, y2) boxes of overlays (score bug, logos) that do NOT
    move with the camera and would otherwise anchor the motion estimate to zero.
    """
    m = np.full(shape[:2], 255, np.uint8)
    for x1, y1, x2, y2 in static_regions:
        m[y1:y2, x1:x2] = 0
    if boxes is not None:
        for x1, y1, x2, y2 in boxes.astype(int):
            pad = 8
            m[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = 0
    return m


def frame_to_frame_h(prev_gray: np.ndarray, gray: np.ndarray, mask: np.ndarray,
                     ) -> tuple[np.ndarray, int]:
    """Homography mapping pixels of `prev` to `gray` from LK optical flow of static points.

    Returns (H, number_of_inliers). Falls back to identity if too few points survive.
    """
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=600, qualityLevel=0.01, minDistance=8,
                                 mask=mask)
    if p0 is None or len(p0) < 12:
        return np.eye(3), 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    if ok.sum() < 12:
        return np.eye(3), 0
    H, inl = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 2.0)
    if H is None:
        return np.eye(3), 0
    return H, int(inl.sum())


def _line_samples(step_m: float = 0.5) -> np.ndarray:
    """Points (pitch metres) every `step_m` along the box and arc markings only."""
    pts = []
    for line in pitch_lines()[:3]:
        for a, b in zip(line[:-1], line[1:]):
            n = max(2, int(np.linalg.norm(b - a) / step_m))
            pts.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    return np.vstack(pts)


def alignment_score(gray: np.ndarray, H_pitch_from_img: np.ndarray, shift_px: int = 18,
                    ) -> float:
    """How well projected markings sit on bright painted lines (higher = better; ~1 = chance).

    Compares the mean white-line response under the projected box/arc lines with the same
    lines shifted down by `shift_px` pixels. Pitch lines are thin and bright, so a
    top-hat filter (bright thin structures) highlights them. A ratio near 1 means the
    overlay is no better than chance; clearly above 1 means it sits on the real lines.
    Only points inside the image count, and fewer than 30 such points returns NaN.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    resp = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k).astype(np.float32)
    resp = cv2.dilate(resp, np.ones((3, 3), np.uint8))   # tolerate ~1-2 px misplacement
    pts = apply_h(np.linalg.inv(H_pitch_from_img), _line_samples())
    h, w = gray.shape

    def mean_resp(p: np.ndarray) -> tuple[float, int]:
        xi, yi = np.round(p[:, 0]).astype(int), np.round(p[:, 1]).astype(int)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        return (float(resp[yi[ok], xi[ok]].mean()) if ok.any() else 0.0), int(ok.sum())

    on, n = mean_resp(pts)
    off, _ = mean_resp(pts + np.array([0, shift_px]))
    return float("nan") if n < 30 else on / max(off, 1e-6)


def chain(Hs_prev_to_next: list[np.ndarray]) -> list[np.ndarray]:
    """Cumulative maps from the reference frame to each later frame.

    Input element k maps frame (ref+k) -> (ref+k+1). Output[0] is the identity (the reference
    itself) and Output[k] maps reference pixels -> frame (ref+k) pixels.
    """
    out = [np.eye(3)]
    for H in Hs_prev_to_next:
        out.append(H @ out[-1])
    return out


def feet_points(boxes: np.ndarray) -> np.ndarray:
    """Bottom-centre of each (x1, y1, x2, y2) box: where the player touches the ground."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    return np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]
