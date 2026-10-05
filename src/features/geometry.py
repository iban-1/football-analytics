"""Shot geometry in StatsBomb coordinates (120 x 80, attacking toward x = 120).

Units are StatsBomb pitch units (roughly yards: 120 x 80 maps to 105 x 68 m).
The goal mouth is 8 units wide, posts at y = 36 and y = 44.
"""
from __future__ import annotations

import numpy as np

from src.utils.coords import SB_GOAL_X, SB_GOAL_Y, SB_POST_Y

LEFT_POST = np.array([SB_GOAL_X, SB_POST_Y[0]])
RIGHT_POST = np.array([SB_GOAL_X, SB_POST_Y[1]])


def shot_distance(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
    """Straight-line distance from the shot location to the centre of the goal."""
    return np.hypot(SB_GOAL_X - np.asarray(x, float), SB_GOAL_Y - np.asarray(y, float))


def shot_angle(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
    """Angle (radians) subtended by the two goal posts at the shot location.

    Why this and not the angle to the goal centre: it measures how much of the
    goal the shooter can actually "see". It is the angle between the vectors
    pointing to the left and right post. Wide shots give a tiny angle even if
    the distance is short; directly in front of the goal gives the largest.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ax, ay = LEFT_POST[0] - x, LEFT_POST[1] - y
    bx, by = RIGHT_POST[0] - x, RIGHT_POST[1] - y
    # atan2(|cross|, dot) is numerically stabler than arccos(dot / norms).
    return np.arctan2(np.abs(ax * by - ay * bx), ax * bx + ay * by)


def in_shot_cone(
    px: np.ndarray, py: np.ndarray, x: float, y: float,
) -> np.ndarray:
    """True for points inside the triangle (shot location, left post, right post).

    Uses the sign-of-cross-product test: a point is inside a triangle if it is
    on the same side of all three edges.
    """
    def side(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    px = np.asarray(px, float)
    py = np.asarray(py, float)
    d1 = side(x, y, *LEFT_POST, px, py)
    d2 = side(*LEFT_POST, *RIGHT_POST, px, py)
    d3 = side(*RIGHT_POST, x, y, px, py)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)
