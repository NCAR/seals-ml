import numpy as np

from scipy.interpolate import griddata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestRegressor

## Need a few functuons for the baseline ML, since the inputs are different enough did not put them in the class

# This is not an interpolator
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



### Everything below here should just be an interpolator ###
class ScipyInterpolate(object):
    """
    A scikit-learn compatible class for performing 2D interpolation using scipy's griddata and finding the global maximum.

    Parameters:
    - method (str, optional): Interpolation method, default is 'cubic'.
    """

    def __init__(self, method="cubic"):
        self.method = method

    def fit(self, x_test, y_test):
        """
        Stores the sensor data for later use in the predict method.

        Parameters:
        - x (array-like): X-coordinates of the sensor data.
        - y (array-like): Y-coordinates of the sensor data.
        - z (array-like): Sensor values corresponding to (x, y).

        Returns:
        - self (ScipyInterpolate): Fitted instance.
        """
        self.x_sensors_ = x_test[:,0]
        self.y_sensors_ = x_test[:,1]
        self.z_sensors_ = y_test
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
        from scipy.interpolate import griddata

        z_interpolated = griddata((self.x_sensors_, self.y_sensors_), self.z_sensors_, (x_test), method=self.method)
        z_interpolated = z_interpolated.reshape(output_shape)

        return z_interpolated
  
class GaussianProcessInterpolator():
    """
    A scikit-learn compatible class for performing Gaussian Process interpolation.

    Parameters:
    - length_scale (float, optional): Length scale for the RBF kernel. Default is 10.
    """

    def __init__(self, length_scale=10):
        self.length_scale = length_scale

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
        self.gp_model = GaussianProcessRegressor(kernel=RBF(length_scale=self.length_scale))
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

    def predict(self, x_test, output_shape = (100,100)):
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
        self.x_test_ = x_test
        self.output_shape_ = output_shape

        # Predict interpolated values
        interpolated_values = self.rf_model.predict(self.x_test_)

        # Reshape interpolated values to match the mesh dimensions
        interpolated_values_rf = interpolated_values.reshape(self.output_shape_)

        return interpolated_values_rf