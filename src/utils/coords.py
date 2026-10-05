"""Coordinate conversion helpers.

StatsBomb: pitch is 120 x 80 units, origin top-left, x grows toward the
attacking goal (always left-to-right for the team in possession), y grows
downward. The goal centre is at (120, 40) and the posts at y = 36 and 44
(goal width 8 units = 7.32 m on a real pitch).

Metrica: normalised 0-1 with (0, 0) top-left, (1, 1) bottom-right.
"""
from __future__ import annotations

import numpy as np

SB_LENGTH = 120.0
SB_WIDTH = 80.0
SB_GOAL_X = 120.0
SB_GOAL_Y = 40.0
SB_POST_Y = (36.0, 44.0)


def statsbomb_to_metres(
    x: np.ndarray | float, y: np.ndarray | float,
    length_m: float = 105.0, width_m: float = 68.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rescale StatsBomb units to metres (assumed pitch size, default 105x68)."""
    return np.asarray(x) * length_m / SB_LENGTH, np.asarray(y) * width_m / SB_WIDTH


def metrica_to_metres(
    x: np.ndarray | float, y: np.ndarray | float,
    length_m: float = 105.0, width_m: float = 68.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Metrica normalised (0-1) coordinates to metres."""
    return np.asarray(x) * length_m, np.asarray(y) * width_m


def metrica_to_statsbomb(
    x: np.ndarray | float, y: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Metrica normalised coordinates to StatsBomb 120x80 units."""
    return np.asarray(x) * SB_LENGTH, np.asarray(y) * SB_WIDTH


def flip_statsbomb(
    x: np.ndarray | float, y: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate 180 degrees: view an action from the opposite team's direction."""
    return SB_LENGTH - np.asarray(x), SB_WIDTH - np.asarray(y)
