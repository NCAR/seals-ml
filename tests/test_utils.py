import pytest
import numpy as np
from sealsml import geometry

def test_distance_between_points_3d():
    """
    Tests the `distance_between_points_3d` function.
    """
   # Test case 1: Check distance between two identical points (should be 0)
    point1 = np.array([[0.0, 0.0, 0.0]])
    point2 = np.array([[0.0, 0.0, 0.0]])
    geometry_class = geometry.geo(array1= point1 , array2=point2)
    result = geometry_class.distance_between_points_3d()
    assert np.array_equal(result, np.array([0.0]))

    point1 = np.array([[0.0, 0.0, 0.0]])
    point2 = np.array([[0.0, 1.0, 0.0]])  # Should have distance of 1
    geometry_class = geometry.geo(array1= point1 , array2=point2)
    result = geometry_class.distance_between_points_3d(grid_resolution=1)
    assert np.array_equal(result, np.array([1.]))


def test_calculate_azimuth():
    '''
    Tests the `calculate_azimuth` function.
    '''
    # Test that the function works when the points are valid.
    point1 = np.array([0.0, 0.0, 0.0])
    point2 = np.array([0.0, 1.0, 0.0])
    geometry_class = geometry.geo(array1= point1 , array2=point2)
    result = geometry_class.calculate_azimuth()
    assert np.array_equal(result, 0.0)

    # Test case 2: Check azimuth for points with known azimuth values
    point1 = np.array([0.0, 0.0, 0.0])
    point2 = np.array([1.0, 0.0, 0.0])  # Should have azimuth of 90 degrees
    geometry_class = geometry.geo(array1= point1 , array2=point2)
    result = geometry_class.calculate_azimuth()
    assert np.array_equal(result, 90.0)

    # Test case 3: Check azimuth for 45's
    point1 = np.array([0.0, 0.0, 0.0])
    point2 = np.array([[2,  2,   0],
                       [4,  4,  -3],
                       [12, 12,  2]]) 
    geometry_class = geometry.geo(array1= point1 , array2=point2)
    result = geometry_class.calculate_azimuth()
    assert np.array_equal(result, [45., 45., 45.])

    # Test case 3: Check for an exception when input arrays have different shapes
    point1 = np.array([0.0, 0.0, 0.0])
    point2 = np.array([1.0, 0.0, 0.0, 2.0])  # Different shape
    geometry_class = geometry.geo(array1= point1 , array2=point2)
    with pytest.raises(IndexError):
        result = geometry_class.calculate_azimuth()

def test_dip():
    """Tests the `dip` function."""

    # Test that the function works when the points are valid.
    point1 = np.array([0, 0, 0])
    point2 = np.array([0, 0, 0])
    geometry_class = geometry.geo(array1= point1 , array2=point2)
    result = geometry_class.calculate_elevation_angle()
    assert np.allclose(result, 0)  # Use np.allclose for floating-point comparisons

    # Test that the function raises an error when the points are not valid.
    point3 = np.array([0, 0, 1])
    point4 = np.array([1, 0, 0, 4])
    geometry_class = geometry.geo(array1= point3 , array2=point4)
    with pytest.raises(IndexError):
        result = geometry_class.calculate_elevation_angle()

# the end