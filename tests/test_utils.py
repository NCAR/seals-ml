import os
import pytest
import numpy as np
import xarray as xr
import yaml 

# Our functiuons 
from sealsml.geometry import GeoCalculator, polar_to_cartesian
from sealsml.data import DataSampler, Preprocessor
from sealsml.baseline import GPModel
from sealsml.evaluate import calculate_distance_matrix
from sealsml.staticinference import specific_site_data_generation, extract_ts_segments
from bridgescaler import save_scaler

def test_polar_to_cart1():
    # Test with single values
    distance = 2.0
    ref_azi_sin = 0.5
    ref_azi_cos = np.sqrt(3) / 2

    x, y = polar_to_cartesian(distance, ref_azi_sin, ref_azi_cos)

    assert np.isclose(x, distance * ref_azi_cos, rtol=1e-4)
    assert np.isclose(y, distance * ref_azi_sin, rtol=1e-4)

    # Test with arrays
    distance2 = np.array([1.0, 2.0, 3.0])
    ref_azi_sin2 = np.array([0.0, 0.5, 1.0])
    ref_azi_cos2 = np.array([1.0, np.sqrt(3) / 2, 0.5])

    x, y = polar_to_cartesian(distance2, ref_azi_sin2, ref_azi_cos2)

    np.testing.assert_allclose(x, distance2 * ref_azi_cos2, rtol=1e-6)
    np.testing.assert_allclose(y, distance2 * ref_azi_sin2, rtol=1e-6)

def test_GPModel():
    """
    Tests the `GPModel` function.

    This function should have a scikit-learn interface

    """
    # Test Case #1, make sure that .fit reads in numpy arrays
    rand1 = np.random.rand(1, 99)
    rand2 = np.random.rand(1, 27)
    model = GPModel()
    model.fit(x=(rand1, rand2), y=None)

    # Test Case #2
    test_data_path = os.path.join(os.path.dirname(__file__), '../test_data/training_data_SBL2m_Ug2p5_src1-8kg_b.5.nc')
    test_data = os.path.expanduser(test_data_path)
    assert os.path.exists(test_data), f"File not found: {test_data}"

    # Open up the netCDF using xarray
    data = xr.open_dataset(test_data)
    encoder = data.encoder_input.values[..., 0]
    decoder = data.decoder_input.values[:, :, 0, :, 0]

    predictions = model.predict(x=(encoder, decoder))
    # Encoder shape
    assert(encoder.shape[0] == predictions.shape[0])
    assert(encoder.shape[0] == predictions.sum())
    # Decoder shape
    assert(decoder.shape[:2] == predictions.shape[:2])

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

    sampler = DataSampler(min_trace_sensors=4, max_trace_sensors=12, min_leak_loc=1, max_leak_loc=11, 
                          sensor_height_min=1, sensor_height_max=4, leak_height_min=0, leak_height_max=4, 
                          coord_vars=["ref_distance", "ref_azi_sin", "ref_azi_cos", "ref_elv"],
                          met_vars=['u', 'v', 'w'], emission_vars=['q_CH4'])

    test_data_path = os.path.join(os.path.dirname(__file__), '../test_data/CBL2m_Ug2p5_src1-8kg_a.1')
    test_data = os.path.expanduser(test_data_path)
    ds, num_sources = sampler.load_data([test_data])

    for i in range(num_sources):
        sampler.data_extract(ds.isel(srcDim=i))
        
    time_window_size = 20
    samples_per_window = 2
    window_stride = 10

    data = sampler.sample(time_window_size, samples_per_window, window_stride)
    encoder_input, decoder_input, targets = data['encoder_input'], data['decoder_input'], data['target']

    step_size = np.arange(1, sampler.time_steps - time_window_size, window_stride)
    total_samples = samples_per_window * len(step_size)

    assert encoder_input.shape == (total_samples, sampler.max_trace_sensors + 1, time_window_size, len(sampler.variables), 2)
    assert decoder_input.shape == (total_samples, sampler.max_leak_loc, 1, len(sampler.variables), 2)
    assert targets.shape == (total_samples, sampler.max_leak_loc, 1)

    rand_sample = np.random.randint(1, total_samples, 1)[0]
    rand_time_1, rand_time_2 = (np.random.randint(0, time_window_size,  1)[0],
                                np.random.randint(0, time_window_size,  1)[0])
    # assert mask is equal
    assert (encoder_input[rand_sample, :, rand_time_1, :, -1] == encoder_input[rand_sample, :, rand_time_2, :, -1]).all()

def test_distance_matrix_export():
    array = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    result = calculate_distance_matrix(array, export_matrix=True)
    expected_min = np.sqrt(3)
    expected_median = np.sqrt(3)
    expected_max = np.sqrt(12)
    expected_matrix = np.array([[0.        , 1.73205081, 3.46410162],
                                 [1.73205081, 0.        , 1.73205081],
                                 [3.46410162, 1.73205081, 0.        ]])
    
    # Using pytest.approx with 2 decimal places of tolerance
    assert result[0] == pytest.approx(expected_matrix, abs=1e-2)
    assert result[1] == pytest.approx(expected_min, abs=1e-2)
    assert result[2] == pytest.approx(expected_median, abs=1e-2)
    assert result[3] == pytest.approx(expected_max, abs=1e-2)

def test_static():

    test_data_path = os.path.join(os.path.dirname(__file__), '../test_data/inference_example_v1.nc')
    test_data = os.path.expanduser(test_data_path)
    assert os.path.exists(test_data), f"File not found: {test_data}"

    sitemap_path = os.path.join(os.path.dirname(__file__), '../test_data/sitemap_A.nc')
    sitemap = os.path.expanduser(sitemap_path)
    assert os.path.exists(test_data), f"File not found: {sitemap}"

    ds = specific_site_data_generation(test_data, sitemap, time_window_size=100, window_stride=50)
    assert isinstance(ds, xr.Dataset), "The object is not an xarray.Dataset"
    assert isinstance(ds['encoder_output'], xr.DataArray), "The object is not an xarray.DataArray"
    assert isinstance(ds['decoder_output'], xr.DataArray), "The object is not an xarray.DataArray"

    p = Preprocessor()
    # p.load_scaler("/Users/cbecker/Desktop/scaler_2024-05-23_1736.json")
    scaled_encoder, encoder_mask = p.preprocess(ds['encoder'], fit_scaler=True)
    save_scaler(p.scaler, "./scaler.json")
    p.load_scaler("./scaler.json")
    scaled_decoder, decoder_mask = p.preprocess(ds['decoder'], fit_scaler=False)
    assert scaled_encoder.shape == ds['encoder_output'].shape
    assert scaled_decoder.shape == ds['decoder_output'].squeeze().shape
    assert encoder_mask.shape == (ds['encoder_output'].shape[0], ds['encoder_output'].shape[1])

def test_extract_ts_segments():
    # Test case 1: Regular case
    time_series = np.arange(10)
    segment_length = 3
    stride = 2
    expected_indices = np.array([[0, 3], [2, 5], [4, 7], [6, 9]])
    expected_dropped_elements = np.array([9])
    
    start_end_indices, dropped_elements = extract_ts_segments(time_series, segment_length, stride)
    
    assert np.array_equal(start_end_indices, expected_indices), "Test case 1: Start-End Indices do not match"
    assert np.array_equal(dropped_elements, expected_dropped_elements), "Test case 1: Dropped elements do not match"
    
    # Test case 2: No elements dropped
    time_series = np.arange(9)
    segment_length = 3
    stride = 3
    expected_indices = np.array([[0, 3], [3, 6], [6, 9]])
    expected_dropped_elements = np.array([])
    
    start_end_indices, dropped_elements = extract_ts_segments(time_series, segment_length, stride)
    
    assert np.array_equal(start_end_indices, expected_indices), "Test case 2: Start-End Indices do not match"
    assert np.array_equal(dropped_elements, expected_dropped_elements), "Test case 2: Dropped elements do not match"