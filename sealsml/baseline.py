import numpy as np
import matplotlib.pyplot as plt

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

# Everything below here should be an interpolator
class ScipyInterpolate(object):
    """
    A scikit-learn compatible class for performing 2D interpolation using scipy's griddata and finding the global maximum.

    Parameters:
    - method (str, optional): Interpolation method, default is 'cubic'.
    """

    def __init__(self, method="cubic"):
        self.method = method

    def fit(self, x, y, z):
        """
        Stores the sensor data for later use in the predict method.

        Parameters:
        - x (array-like): X-coordinates of the sensor data.
        - y (array-like): Y-coordinates of the sensor data.
        - z (array-like): Sensor values corresponding to (x, y).

        Returns:
        - self (ScipyInterpolate): Fitted instance.
        """
        self.x_sensors_ = x
        self.y_sensors_ = y
        self.z_sensors_ = z
        return self

    def predict(self, x_mesh, y_mesh):
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

        z_interpolated = griddata((self.x_sensors_, self.y_sensors_), self.z_sensors_, (x_mesh, y_mesh), method=self.method)
        max_z = np.nanmax(z_interpolated)
        max_indices = np.where(z_interpolated == max_z)

        return z_interpolated, max_z, max_indices
  
class GaussianProcessInterpolator():
    """
    A scikit-learn compatible class for performing Gaussian Process interpolation.

    Parameters:
    - length_scale (float, optional): Length scale for the RBF kernel. Default is 10.
    """

    def __init__(self, length_scale=10):
        self.length_scale = length_scale

    def fit(self, x, y, z):
        """
        Fits the Gaussian process model with the sensor data.

        Parameters:
        - x_sensors (array-like): X-coordinates of the sensor data.
        - y_sensors (array-like): Y-coordinates of the sensor data.
        - z_sensors (array-like): Sensor values corresponding to (x_sensors, y_sensors).

        Returns:
        - self (GaussianProcessInterpolator): Fitted instance.
        """
        self.x_sensors_ = x
        self.y_sensors_ = y
        self.z_sensors_ = z

        # Combine sensor data into training features
        x_train = np.column_stack((self.x_sensors_, self.y_sensors_))



        # Fit the Gaussian process model
        self.gp_model = GaussianProcessRegressor(kernel=RBF(length_scale=self.length_scale))
        self.gp_model.fit(x_train, self.z_sensors_)

        return self

    def predict(self, x_mesh, y_mesh):
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
        x_test = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))
        print('shape of x_test', np.shape(x_test))
        # Predict interpolated values
        interpolated_values = self.gp_model.predict(x_test)

        # Reshape interpolated values to match the mesh dimensions
        interpolated_values = interpolated_values.reshape(x_mesh.shape)

        # Finding Global Max and its indices using the Gaussian process model
        max_z = np.nanmax(interpolated_values)
        max_indices = np.where(max_z == interpolated_values)

        return interpolated_values, max_z, max_indices

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

    def fit(self, x, y, z):
        """
        Fits the Random Forest model with the sensor data.

        Parameters:
        - x_sensors (array-like): X-coordinates of the sensor data.
        - y_sensors (array-like): Y-coordinates of the sensor data.
        - z_sensors (array-like): Sensor values corresponding to (x_sensors, y_sensors).

        Returns:
        - self (RandomForestInterpolator): Fitted instance.
        """

        self.x_sensors_ = x
        self.y_sensors_ = y
        self.z_sensors_ = z


        # Combine sensor data into training features
        x_train = np.column_stack((self.x_sensors_, self.y_sensors_))

        # Create the Random Forest Regressor
        self.rf_model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state)

        # Fit the model with the data
        self.rf_model.fit(x_train, self.z_sensors_)

        return self

    def predict(self, x_mesh, y_mesh):
        """
        Performs interpolation on the provided mesh coordinates using the fitted Random Forest model.

        Parameters:
        - x_mesh (array-like): X-coordinates for interpolation.
        - y_mesh (array-like): Y-coordinates for interpolation.

        Returns:
        - interpolated_values (ndarray): Interpolated values at the specified coordinates.
        - max_z (float): Global maximum value of the interpolated data.
        - max_indices (tuple): Indices of the global maximum in the interpolated data.
        """
        # Combine mesh coordinates into test features
        x_test = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))

        # Predict interpolated values
        interpolated_values_rf = self.rf_model.predict(x_test)

        # Reshape interpolated values to match the mesh dimensions
        interpolated_values_rf = interpolated_values_rf.reshape(x_mesh.shape)

        # Finding Global Max and its indices using the Random Forest model
        max_z_rf = np.nanmax(interpolated_values_rf)
        max_indices_rf = np.unravel_index(np.argmax(interpolated_values_rf), interpolated_values_rf.shape)

        return interpolated_values_rf, max_z_rf, max_indices_rf
    
