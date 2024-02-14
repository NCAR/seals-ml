# Standard library
import sys
import os

# Data manipulation and analysis
import numpy as np
import xarray as xr

# Scientific computing and machine learning
from scipy.interpolate import griddata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import MinMaxScaler
from sealsml.geometry import polar_to_cartesian

## Need a few functuons for the baseline ML, since the inputs are different enough did not put them in the class


def create_meshgrid(x_sensors, y_sensors, buffer=6, grid_points=100):
    """
    Create a meshgrid based on sensor positions.

    Parameters:
    - x_sensors (array-like): x-coordinates of sensor positions.
    - y_sensors (array-like): y-coordinates of sensor positions.
    - buffer (float, optional): Buffer value added to the minimum and maximum coordinates for each axis.
    - grid_points (int, optional): Number of grid points in each dimension.

    Returns:
    - x_new (ndarray): Meshgrid for x-coordinates.
    - y_new (ndarray): Meshgrid for y-coordinates.
    """
    x_new, y_new = np.mgrid[min(x_sensors)-buffer:max(x_sensors)+buffer:grid_points*1j, min(y_sensors)-buffer:max(y_sensors)+buffer:grid_points*1j]
    return x_new, y_new

def nanargmax_to_one(arr):
    # Find the index of the maximum non-NaN value
    max_index = np.nanargmax(arr)

    # Create a new array with 1 at the max_index and 0 elsewhere
    result_array = np.zeros_like(arr)
    result_array[max_index] = 1

    return result_array

def find_closest_values_with_indices(arr1, arr2):
    """
    Find the closest values in arr2 for each element in arr1 and return their values and indices.

    Parameters:
    - arr1 (list or array-like): The first array containing elements for which the closest values are sought.
    - arr2 (list or array-like): The second array from which the closest values are selected.

    Returns:
    tuple: A tuple containing two elements:
        - closest_values (list): A list of closest values for each element in arr1.
        - closest_indices (list): A list of indices corresponding to the closest values in arr2.
    """
    # Convert input lists to NumPy arrays for better performance
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    # Initialize lists to store the closest values and their indices
    closest_values = []
    closest_indices = []

    # Check if arr2 is empty
    if len(arr2) == 0:
        return closest_values, closest_indices

    # Iterate over each element in arr1
    for val1 in arr1:
        # Calculate absolute differences for all elements in arr2
        differences = np.abs(arr2 - val1)

        # Find the index with the minimum difference
        min_diff_index = np.argmin(differences)

        # Append the closest value and its index to the respective lists
        closest_values.append(arr2[min_diff_index])
        closest_indices.append(min_diff_index)

    # Return the lists of closest values and indices
    return closest_values, closest_indices

def remove_all_rows_with_val(arr, value_to_drop=0):
    """
    Removes rows containing all zeros from a NumPy array.

    Parameters:
    - arr (numpy.ndarray): The input array from which rows with all zeros will be removed.

    Returns:
    numpy.ndarray: An array with rows containing all zeros removed.

    Raises:
    TypeError: If the input is not a NumPy array.
    """
    # Check if the input is a numpy array
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a numpy array")

    # Check if the array is empty
    if arr.size == 0:
        return arr

    # Find rows where all elements are zero
    non_zero_rows = np.any(arr != value_to_drop, axis=1)

    # Filter the array to keep only non-zero rows
    result = arr[non_zero_rows]

    return result



class GPModel():

    def __init__(self, num_met_sensors=1, normalize_y=True, n_restarts_optimizer=16, length_scale_bounds=(1e-02, 1e02)):

        self.num_met_sensors = num_met_sensors
        self.normalize_y = normalize_y 
        self.n_restarts_optimizer = n_restarts_optimizer
        self.length_scale_bounds = length_scale_bounds

    def fit(self, x, y, **kwargs):
        '''
        Due to how the interpolator works, there is no 'fit' needed ie a training dataset. 
        
        This fit function just checks that both inputs are numpy arrays.
        '''
        print('fit does nothing')
        self.e_ = x[0]
        self.d_ = x[1]
       
        if not isinstance(self.e_, np.ndarray) or not isinstance(self.d_, np.ndarray):
            raise TypeError("Inputs must be NumPy arrays")
        
    def predict(self, x, batch_size=None):
        '''
        This function uses numpy arrays as input

        x: same shape as encoder.values where each dimension is sample, sensor, time, variable ,mask 
        y: same shape as decoder.values where each dimension is sample, pot_leak, target_time, variable, mask

        The variable should have a length of 8: ['ref_distance', 'ref_azi_sin', 'ref_azi_cos', 'ref_elv', 'u', 'v', 'w', 'q_CH4']
        
        This model does not use any wind information (u,v,w)
        '''
        self.encoder = x[0]
        self.decoder = x[1]

        # Create empty lists
        leak_loc = []

        # This is just used for debugging. Since it's outside of the loop, should not affect runtime that much.
        print('encoder shape (x)', np.shape(self.encoder))
        print('decoder shape (y)', np.shape(self.decoder))
        
        # Collapse on time. Time steps should be constant for this model. 
        max_time = np.max(self.encoder, axis=2) # this can be changed to 80th percentile or w/e

        # make xarray datasets
        dataset = xr.Dataset({"data": (["sample", "sensor", "variables"], max_time)})
        decoder_dataset = xr.Dataset({"data": (["sample", "leak_locations", "variables"], self.decoder)})
    
        for i in dataset.sample:
            # pulling the information from the sensors, distance, azi_sin and azi_cos and ch4
            ref_dist = dataset.isel(sample=i).data.values[:, 0]
            azi_sin  = dataset.isel(sample=i).data.values[:, 1]
            azi_cos  = dataset.isel(sample=i).data.values[:, 2]
            ch4_data = dataset.isel(sample=i).data.values[:, 7]

            combined_sensor = np.column_stack((ref_dist, azi_sin, azi_cos, ch4_data))

            # Number of met sensors is set constant here. This is one spot where some more clever logic might be needed
            combined_sensor = combined_sensor[self.num_met_sensors:] # drops the met sensor, in this case it's the first row. 
            drop_masked_sensor =  remove_all_rows_with_val(combined_sensor, -1) # This line basically up-padds to the correct number of sensors
            
            x_, y_ = polar_to_cartesian(drop_masked_sensor[:, 0], drop_masked_sensor[:, 1], drop_masked_sensor[:, 2])
            ch4_data = drop_masked_sensor[:, 3]

            # Same workflow, now for (potential) leak locations
            ref_dist_leak = decoder_dataset.isel(sample=i).data.values[:, 0]
            azi_sin_leak  = decoder_dataset.isel(sample=i).data.values[:, 1]
            azi_cos_leak  = decoder_dataset.isel(sample=i).data.values[:, 2]

            combined_leak_loc = np.column_stack((ref_dist_leak, azi_sin_leak, azi_cos_leak))
            drop_masked_leaks =  remove_all_rows_with_val(combined_leak_loc, -1) # this drops all padded leaks

            x_leaks, y_leaks = polar_to_cartesian(drop_masked_leaks[:, 0], drop_masked_leaks[:, 1], drop_masked_leaks[:, 2])

            # At this point, we have x,y and (median) ch4 for sensors, and x and y locations for potential leak locations
           
            # X and Y of sensors in cartesian coordiantes
            X_train = np.column_stack((x_.ravel(), y_.ravel()))

            # CH4 data from those sensors
            y_train = ch4_data

            # X and Y locations of the potential leaks
            X_test = np.column_stack((x_leaks.ravel(), y_leaks.ravel()))

            ## Make the model
            kernel = 1.0 * RBF(length_scale_bounds=self.length_scale_bounds)
            
            gp_model = GaussianProcessRegressor(kernel=kernel,
                                                n_restarts_optimizer=self.n_restarts_optimizer,
                                                normalize_y=self.normalize_y)
        
            gp_model.fit(X_train, y_train)
            gp_results = gp_model.predict(X_test)

            # need to pad to 20 to match leak locations
            padded_array = np.pad(nanargmax_to_one(gp_results),
                                  (0, decoder_dataset.leak_locations.size - len(nanargmax_to_one(gp_results))), mode='constant')

            leak_loc.append(padded_array)

        return np.asarray(leak_loc)  


#### Everything below here should just be an interpolator ####
# The below is useful if we want a grided output. 
    

class ScipyInterpolate(object):
    """
    A scikit-learn compatible class for performing 2D interpolation using scipy's griddata and finding the global maximum.

    Parameters:
    - method (str, optional): Interpolation method, default is 'cubic'.
    """

    def __init__(self, method="cubic"):
        self.method = method

    def fit(self, x_train, y_train):
        """
        Stores the sensor data for later use in the predict method.

        Parameters:
        - x (array-like): X-coordinates of the sensor data.
        - y (array-like): Y-coordinates of the sensor data.
        - z (array-like): Sensor values corresponding to (x, y).

        Returns:
        - self (ScipyInterpolate): Fitted instance.
        """
        self.x_sensors_ = x_train[:,0]
        self.y_sensors_ = x_train[:,1]
        self.z_sensors_ = y_train
        return self

    def predict(self, x_test, output_shape = (100,100)):
        """
        Performs interpolation and finds the global maximum on the provided mesh coordinates using the stored sensor data.

        Parameters:
        - x_mesh (array-like): X-coordinates for which to interpolate.
        - y_mesh (array-like): Y-coordinates for which to interpolate.

        Returns:
        - z_interpolated (ndarray): Interpolated values at the specified coordinates.
        - max_z (float): Global maximum value of the interpolated data.
        - max_indices (tuple): Indices of the global maximum in the interpolated data.

        Note:
        - The function uses the interpolation method specified during initialization.
        """
        z_interpolated = griddata((self.x_sensors_, self.y_sensors_), self.z_sensors_, (x_test), method=self.method)
        z_interpolated = z_interpolated.reshape(output_shape)

        return z_interpolated
  
class GaussianProcessInterpolator():
    """
    A scikit-learn compatible class for performing Gaussian Process interpolation.

    Parameters:
    - length_scale (float, optional): Length scale for the RBF kernel. Default is 10.
    """

    def __init__(self, length_scale=5, n_restarts_optimizer=3, normalize_y=False):
        self.length_scale = length_scale
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y

    def fit(self, x_train, y_train):
        """
        Fits the Gaussian process model with the sensor data.

        Parameters:
        - X Test (made up of X & Y locations) x_sensors (array-like): X-coordinates of the sensor data.
        - Y_test: (array-like): Sensor values corresponding to X_test(x_sensors, y_sensors).

        Returns:
        - self (GaussianProcessInterpolator): Fitted instance.
        """
        self.x_train_ = x_train
        self.y_train_ = y_train

        # Fit the Gaussian process model
        kernel = 1.0 * RBF(length_scale_bounds=(1e-01, 1e02))
        self.gp_model = GaussianProcessRegressor(kernel=kernel,
                                                n_restarts_optimizer=self.n_restarts_optimizer,
                                                normalize_y = self.normalize_y)
        
        self.gp_model.fit(x_train, self.y_train_)

        return self

    def predict(self, x_test, output_shape = (100,100)):
        """
        Performs interpolation on the provided mesh coordinates using the fitted Gaussian Process model.

        Parameters:
        - x_mesh (array-like): X-coordinates for interpolation.
        - y_mesh (array-like): Y-coordinates for interpolation.

        Returns:
        - interpolated_values (ndarray): Interpolated values at the specified coordinates.
        - max_z (float): Global maximum value of the interpolated data.
        - max_indices (tuple): Indices of the global maximum in the interpolated data.
        """
        # Combine mesh coordinates into test features
        self.x_test_ = x_test
    
        # Predict interpolated values
        interpolated_values = self.gp_model.predict(self.x_test_)

        # Reshape interpolated values to match the mesh dimensions
        interpolated_values = interpolated_values.reshape(output_shape)

        return interpolated_values

class RandomForestInterpolator():
    """
    A scikit-learn compatible class for performing Random Forest interpolation.

    Parameters:
    - max_depth (int, optional): Maximum depth of the decision trees. Default is 2.
    - n_estimators (int, optional): Number of trees in the forest. Default is 50.
    - random_state (int, optional): Seed for random number generation. Default is 42.
    """

    def __init__(self, max_depth=2, n_estimators=50, random_state=42):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X_train, y_train):
        """
        Fits the Random Forest model with the sensor data.

        Parameters:
        - x_sensors (array-like): X-coordinates of the sensor data.
        - y_sensors (array-like): Y-coordinates of the sensor data.
        - z_sensors (array-like): Sensor values corresponding to (x_sensors, y_sensors).

        Returns:
        - self (RandomForestInterpolator): Fitted instance.
        """
        self.x_train_ = X_train
        self.y_train_ = y_train

        # Create the Random Forest Regressor
        self.rf_model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state)

        # Fit the model with the data
        self.rf_model.fit(self.x_train_, self.y_train_)

        return self

    def predict(self, x, output_shape = (100,100)):
        """
        Performs interpolation on the provided mesh coordinates using the fitted Random Forest model.

        Parameters:
        - x_text:
        - output_shape:

        Returns:
        - interpolated_values (ndarray): Interpolated values at the specified coordinates.
        - max_z (float): Global maximum value of the interpolated data.
        - max_indices (tuple): Indices of the global maximum in the interpolated data.
        """
        self.x_test_ = x
        self.output_shape_ = output_shape

        # Predict interpolated values
        interpolated_values = self.rf_model.predict(self.x_test_)

        # Reshape interpolated values to match the mesh dimensions
        interpolated_values_rf = interpolated_values.reshape(self.output_shape_)

        return interpolated_values_rf

# This will probably be deleted at a later date 
    
def gaussian_interp(data, mesh_dim=30):
    '''
    Input is a Charlie-sampled dataset. Needs the meta, and not just the encoder/decoder. 
    
    Input is netCDF file direct which gets convered to xarray dataset.

    '''
    
    if not data.lower().endswith(".nc"):  # Case-insensitive check
        raise ValueError("Input file must be a NetCDF file with a .nc extension.")

    ds = xr.open_dataset(data)
    
    num_sensor_max = ds.dims['sensor']
    reshape_val = ds.dims['sample'] * ds.dims['sensor'] 

    # Create empty lists
    all_reshaped_gp_results = []
    leak_loc = []
    leak_number = []

    number_of_sensors = []
    number_of_leaks = []
    sample_number = []
    pred_ch4 = []
    sensor_ch4 = []
    
    reshaped  = np.reshape(ds.sensor_meta.values, (reshape_val ,3))
    all_sensor_loc = remove_all_rows_with_val(reshaped)

    ## Making one grid for the entire file ## 
    x_new, y_new = create_meshgrid(all_sensor_loc[:,0], all_sensor_loc[:,1], buffer=2, grid_points=mesh_dim)
    
    for i in ds.sample.values:
        # print('sample number', i)

        sensor_locations = remove_all_rows_with_val(ds.sensor_meta.isel(sample=i).values)
        num_of_sensors = sensor_locations.shape[0]

        leak_locations = remove_all_rows_with_val(ds.leak_meta.isel(sample=i).values)
        num_of_leaks = leak_locations.shape[0]

        ch4_data = ds.encoder_input.isel(sample=i, sensor=slice(0, num_of_sensors), mask=0).sel(variable='q_CH4').values
        # We are going to take the median. Could also take the P80, etc. Most sensors are either on or off.
        ch4_median = np.median(ch4_data, axis=1)
    
        # new mesh data points
        X_test = np.column_stack((x_new.ravel(), y_new.ravel()))

        X_train = np.column_stack((sensor_locations[:,0], sensor_locations[:,1]))
        y_train = ch4_median

        ## Make the model
        gp_mo = GaussianProcessInterpolator(length_scale=10, n_restarts_optimizer=10, normalize_y=True) # this needs to be small to not barf
        gp_mo.fit(X_train, y_train)

        # Fit it - Interpolated Results
        reshaped_gp_results = gp_mo.predict(X_test, output_shape = (mesh_dim,mesh_dim))

        # Let's find the leak locations, and then mark that with a 1
        closest_values_x, indicies_x = find_closest_values_with_indices(leak_locations[:,0], x_new.diagonal())
        closest_values_y, indicies_y = find_closest_values_with_indices(leak_locations[:,1], y_new.diagonal())
        gp_    = reshaped_gp_results[indicies_x, indicies_y]
    
        # need to pad to 20 to match leak locations
        padded_array = np.pad(nanargmax_to_one(gp_), (0, 20 - len(nanargmax_to_one(gp_))), mode='constant')
        padded_ch4 =   np.pad((gp_), (0, 20 - len(nanargmax_to_one(gp_))), mode='constant')

        # let's also store what sensor is leaking:
        where_padded_one = np.where(padded_array == 1)

        # Let's store sensor median values, padded to max number of sensros (10)
        padded_sensor_ch4 = np.pad(ch4_median, (0, 10 - len(ch4_median)), mode='constant')

        # append it
        sample_number.append(i) 

        number_of_sensors.append(num_of_sensors)
        number_of_leaks.append(num_of_leaks)
    
        sensor_ch4.append(padded_sensor_ch4)

        leak_number.append(np.asarray(where_padded_one)[0][0])
        pred_ch4.append(padded_ch4)

        leak_loc.append(padded_array)
        all_reshaped_gp_results.append(reshaped_gp_results)
    # no longer in the loop
    ## Make an xarray results file ##

    # Define coordinates
    coords = {'SampleNumber': sample_number}

    # Create DataArrays
    number_of_sensors_da = xr.DataArray(number_of_sensors, coords=[('SampleNumber', sample_number)], name='NumberOfSensors')
    number_of_leaks_da = xr.DataArray(number_of_leaks, coords=[('SampleNumber', sample_number)], name='NumberOfLeaks')
    leak_number_da = xr.DataArray(leak_number, coords=[('SampleNumber', sample_number)], name='LeakNumber')

    leak_loc_da = xr.DataArray(
        np.asarray(leak_loc),
        coords={'SampleNumber': sample_number, 'MaxNumLeaks': np.arange(np.asarray(leak_loc).shape[1])},
        name='leak_loc'
    )

    ch4_pred_da = xr.DataArray(
        np.asarray(pred_ch4),
        coords={'SampleNumber': sample_number, 'MaxNumLeaks': np.arange(np.asarray(leak_loc).shape[1])},
        name='ch4_at_each_leak_location'
    )

    sensor_ch4_da = xr.DataArray(
        np.asarray(sensor_ch4),
        coords={'SampleNumber': sample_number, 'MaxNumSensors': np.arange(num_sensor_max)},
        name='median_ch4_at_each_sensor'
    )

    interpolation_da = xr.DataArray(
        np.asarray(all_reshaped_gp_results),
        coords={'SampleNumber': sample_number, 'x': x_new.diagonal(), 'y': y_new.diagonal()},
        name='GaussianProcessesInterpolation'
    )

    dataset = xr.Dataset(
        {'NumberOfSensors': number_of_sensors_da, 
        'SensorMedian_ch4': sensor_ch4_da,
        'NumberOfLeaks': number_of_leaks_da, 
        'LeakNumber': leak_number_da,
        'PredLeakLocation': leak_loc_da,
        'Predch4value': ch4_pred_da,
        'interpolation': interpolation_da},
        coords=coords
    )

    return dataset 