import numpy as np

from src.utils.coords import (flip_statsbomb, metrica_to_metres,
                              metrica_to_statsbomb, statsbomb_to_metres)


def test_statsbomb_to_metres_corners_and_centre():
    x, y = statsbomb_to_metres(np.array([0, 60, 120]), np.array([0, 40, 80]))
    assert np.allclose(x, [0, 52.5, 105])
    assert np.allclose(y, [0, 34, 68])


def test_metrica_to_metres_centre():
    x, y = metrica_to_metres(0.5, 0.5)
    assert (x, y) == (52.5, 34.0)


def test_metrica_to_statsbomb_centre_and_corner():
    assert tuple(map(float, metrica_to_statsbomb(0.5, 0.5))) == (60.0, 40.0)
    assert tuple(map(float, metrica_to_statsbomb(1, 1))) == (120.0, 80.0)


def test_flip_is_involution_and_maps_goal_to_goal():
    assert tuple(map(float, flip_statsbomb(120, 40))) == (0.0, 40.0)
    x, y = flip_statsbomb(*flip_statsbomb(33.0, 12.0))
    assert (float(x), float(y)) == (33.0, 12.0)
