import pytest
import numpy as np
from sealsml.utils import distance_between_points_3d, azimuth, dip


def test_distance_between_points_3d():
    """
    Tests the `distance_between_points_3d` function.
    """
    # Test that the function works when the points are valid.
    point1 = np.array([1, 2, 3])
    point2 = np.array([4, 5, 6])
    distance_result = distance_between_points_3d(point1, point2)
    expected_distance = np.sqrt(109)
    assert np.allclose(distance_result, expected_distance)  # Use np.allclose for floating-point comparisons

    # Test that the function raises an error when the points are not valid.
    point1 = np.array([1, 2, 3])
    point2 = np.array([4, 5, 3, 2])
    with pytest.raises(TypeError):
        distance_between_points_3d(point1, point2)

def test_azimuth():
    '''
    Tests the `azimuth` function.
    '''

    # Test that the function works when the points are valid.
    point1 = np.array([1, 0, 0])
    point2 = np.array([1, 1, 0])
    azimuth_result = azimuth(point1, point2)
    assert np.allclose(azimuth_result, 45)  # Use np.allclose for floating-point comparisons

    # Test that the function raises an error when the points are not valid.
    point1 = np.array([1, 0, 0])
    point2 = np.array([1, 2, 0
                       4, 5, 4])
    with pytest.raises(TypeError):
        azimuth(point1, point2)int2)

def test_dip():
    """Tests the `dip` function."""

    # Test that the function works when the points are valid.
    point1 = np.array([0, 0, 1])
    point2 = np.array([0, 0, 2])
    dip_result = dip(point1, point2)
    assert np.allclose(dip_result, 90)  # Use np.allclose for floating-point comparisons

    # Test that the function raises an error when the points are not valid.
    point1 = np.array([0, 0, 1])
    point2 = np.array([1, 0, 0, 4])
    with pytest.raises(TypeError):
        dip(point1, point2)2)

