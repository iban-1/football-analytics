import numpy as np
import pytest

from src.features.geometry import in_shot_cone, shot_angle, shot_distance


def test_distance_from_penalty_spot_and_goal_centre():
    assert shot_distance(108, 40) == pytest.approx(12.0)
    assert shot_distance(120, 40) == pytest.approx(0.0)
    # 3-4-5 triangle: 9 units back, 12 to the side -> hypotenuse 15
    assert shot_distance(111, 28) == pytest.approx(15.0)


def test_angle_from_penalty_spot_hand_checked():
    # vectors (12,-4) and (12,4): angle = 2 * atan(4/12)
    assert shot_angle(108, 40) == pytest.approx(2 * np.arctan(4 / 12))


def test_angle_on_goal_line_between_posts_is_pi():
    assert shot_angle(120, 40) == pytest.approx(np.pi)


def test_angle_on_goal_line_outside_posts_is_zero():
    assert shot_angle(120, 30) == pytest.approx(0.0, abs=1e-9)


def test_angle_shrinks_with_distance_and_width():
    assert shot_angle(100, 40) < shot_angle(110, 40)
    assert shot_angle(110, 20) < shot_angle(110, 40)


def test_in_shot_cone():
    px = np.array([115.0, 115.0, 100.0, 118.0])
    py = np.array([40.0, 60.0, 40.0, 38.0])
    # shooter at (110, 40): (115,40) inside; (115,60) wide; (100,40) behind shooter;
    # (118,38) inside the narrowing triangle
    assert list(in_shot_cone(px, py, 110, 40)) == [True, False, False, True]
