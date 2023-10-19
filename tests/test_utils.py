import pytest
import numpy as np
import xarray as xr
from sealsml.geometry import GeoCalculator
from sealsml.data import DataSampler, Scaler4D

def test_distance_between_points_3d():
    """
    Tests the `distance_between_points_3d` function.
    """
   # Test case 1: Check distance between two identical points (should be 0)
    point1 = np.array([[0.0, 0.0, 0.0]])
    point2 = np.array([[0.0, 0.0, 0.0]])
    geometry_class = GeoCalculator(point1, point2)
    result = geometry_class.distance_between_points_3d()
    assert np.array_equal(result, np.array([0.0]))

    point1 = np.array([[0.0, 0.0, 0.0]])
    point2 = np.array([[0.0, 1.0, 0.0]])  # Should have distance of 1
    geometry_class = GeoCalculator(point1, point2)
    result = geometry_class.distance_between_points_3d(grid_resolution=1)
    assert np.array_equal(result, np.array([1.]))


def test_calculate_azimuth():
    '''
    Tests the `calculate_azimuth` function.
    '''
    # Test that the function works when the points are valid.
    point1 = np.array([0.0, 0.0, 0.0])
    point2 = np.array([0.0, 1.0, 0.0])
    geometry_class = GeoCalculator( point1 , point2)
    result = geometry_class.calculate_azimuth()
    assert np.array_equal(result, 0.0)

    # Test case 2: Check azimuth for points with known azimuth values
    point1 = np.array([0.0, 0.0, 0.0])
    point2 = np.array([1.0, 0.0, 0.0])  # Should have azimuth of 90 degrees
    geometry_class = GeoCalculator( point1 , point2)
    result = geometry_class.calculate_azimuth()
    assert np.array_equal(result, 90.0)

    # Test case 3: Check azimuth for 45's
    point1 = np.array([0.0, 0.0, 0.0])
    point2 = np.array([[2,  2,   0],
                       [4,  4,  -3],
                       [12, 12,  2]]) 
    geometry_class = GeoCalculator(point1 , point2)
    result = geometry_class.calculate_azimuth()
    assert np.array_equal(result, [45., 45., 45.])

    # Test case 3: Check for an exception when input arrays have different shapes
    point1 = np.array([0.0, 0.0, 0.0])
    point2 = np.array([1.0, 0.0, 0.0, 2.0])  # Different shape
    geometry_class = GeoCalculator(point1 , point2)
    with pytest.raises(IndexError):
        result = geometry_class.calculate_azimuth()

def test_dip():
    """Tests the elevation angle (dip) function."""

    # Test that the function works when the points are valid.
    point1 = np.array([0, 0, 0])
    point2 = np.array([0, 0, 0])
    geometry_class = GeoCalculator(point1 , point2)
    result = geometry_class.calculate_elevation_angle()
    assert np.allclose(result, 0)  # Use np.allclose for floating-point comparisons

    # Test that the function works when the points are valid.
    point1 = np.array([0, 0, 0])
    point2 = np.array([0, 1, 1])
    geometry_class = GeoCalculator(point1 , point2)
    result = geometry_class.calculate_elevation_angle()
    assert np.allclose(result, 45)  

    # Test that the function works when the points are valid.
    point1 = np.array([0, 0, 0])
    point2 = np.array([0, 0, 1])
    geometry_class = GeoCalculator(point1 , point2)
    result = geometry_class.calculate_elevation_angle()
    assert np.allclose(result, 90)  

    # Test that the function works when the points are valid.
    point1 = np.array([0, 0, 1])
    point2 = np.array([0, 0, 0])
    geometry_class = GeoCalculator(point1 , point2)
    result = geometry_class.calculate_elevation_angle()
    assert np.allclose(result, -90)  

    # Test that the function raises an error when the points are not valid.
    point3 = np.array([0, 0, 1])
    point4 = np.array([1, 0, 0, 4])
    geometry_class = GeoCalculator(point3 , point4)
    with pytest.raises(IndexError):
        result = geometry_class.calculate_elevation_angle()


def test_DataSampler():

    u = np.random.random(size=(361, 15, 30,  30))
    v = np.random.random(size=(361, 15, 30, 30))
    w = np.random.random(size=(361, 15, 30, 30))
    ch4 = np.random.random(size=(361, 15, 30, 30))
    xPos = np.random.random(size=(15, 30,  30))
    yPos = np.random.random(size=(15, 30, 30))
    zPos = np.random.random(size=(15, 30, 30))
    ref_distance = np.zeros(shape=(15, 30, 30))
    ref_azi_sin = np.zeros(shape=(15, 30, 30))
    ref_azi_cos = np.zeros(shape=(15, 30, 30))
    ref_elv = np.zeros(shape=(15, 30, 30))

    sampler = DataSampler(min_trace_sensors=4, max_trace_sensors=12, min_leak_loc=1, max_leak_loc=11, sensor_height=3,
                          coord_vars=["ref_distance", "ref_azi_sin", "ref_azi_cos", "ref_elv"],
                          met_vars=['u', 'v', 'w'], emission_vars=['q_CH4'])

    sampler.data = xr.Dataset(data_vars=dict(u=(["timeDim", "kDim", "jDim", "iDim"], u),
                                             v=(["timeDim", "kDim", "jDim", "iDim"], v),
                                             w=(["timeDim", "kDim", "jDim", "iDim"], w),
                                             xPos=(["kDim", "jDim", "iDim"], xPos),
                                             yPos=(["kDim", "jDim", "iDim"], yPos),
                                             zPos=(["kDim", "jDim", "iDim"], zPos),
                                             q_CH4=(["time", "kDim", "jDim", "iDim"], ch4),
                                             ref_distance=(["kDim", "jDim", "iDim"], ref_distance),
                                             ref_azi_sin=(["kDim", "jDim", "iDim"], ref_azi_sin),
                                             ref_azi_cos=(["kDim", "jDim", "iDim"], ref_azi_cos),
                                             ref_elv=(["kDim", "jDim", "iDim"], ref_elv)))

    sampler.data = sampler.data.swap_dims({"time": "timeDim"})
    sampler.time_steps = len(sampler.data['timeDim'].values)
    sampler.iDim = len(sampler.data.iDim)
    sampler.jDim = len(sampler.data.jDim)
    sampler.x = np.linspace(0, 58, 30)
    sampler.y = np.linspace(0, 58, 30)
    sampler.z = np.linspace(0, 56, 15)
    time_window_size = 100
    samples_per_window = 2
    window_stride = 50

    data = sampler.sample(time_window_size, samples_per_window, window_stride)
    encoder_input, decoder_input, targets = data['encoder_input'], data['decoder_input'], data['target']

    total_samples = (((sampler.time_steps - time_window_size) // window_stride) + 1) * samples_per_window

    assert encoder_input.shape == (total_samples, sampler.max_trace_sensors, time_window_size, len(sampler.variables), 2)
    assert decoder_input.shape == (total_samples, sampler.max_leak_loc, 1, len(sampler.variables), 2)
    assert targets.shape == (total_samples, sampler.max_leak_loc, 1)

    rand_sample = np.random.randint(1, total_samples, 1)[0]
    rand_time_1, rand_time_2 = np.random.randint(0, 100,  1)[0], np.random.randint(0, 100,  1)[0]
    # assert mask is equal
    assert (encoder_input[rand_sample, :, rand_time_1, :, -1] == encoder_input[rand_sample, :, rand_time_2, :, -1]).all()


def test_scaler():

    scaler = Scaler4D(kind="quantile")
    n_samples = 1000
    n_sensors = 15
    n_leaks = 10
    n_time_steps= 50
    n_variables = 8

    encoder_x = np.random.random(size=(n_samples, n_sensors, n_time_steps, n_variables))
    decoder_x = np.random.random(size=(n_samples, n_leaks, 1, n_variables))

    encoder_transformed = scaler.fit_transform(encoder_x)
    decoder_transformed = scaler.transform(decoder_x)

    assert np.max(encoder_transformed) == 1
    assert np.min(encoder_transformed) == 0
    assert encoder_transformed.shape == (n_samples, n_sensors, n_time_steps * n_variables)
    assert decoder_transformed.shape == (n_samples, n_leaks, 1 * n_variables)
