import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import griddata

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from sklearn.ensemble import RandomForestRegressor

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


class BaselineModels(object):

  def __init__(self, x_sensors, y_sensors, z_sensors, x_mesh, y_mesh):

      self.x_sensors = x_sensors
      self.y_sensors = y_sensors
      self.z_sensors = z_sensors

      self.x_mesh = x_mesh
      self.y_mesh = y_mesh

  def scipy_interpolate(self, method='cubic'):
    """
    Perform 2D interpolation using scipy's griddata and find the global maximum and its indices.

    Parameters:
    - x_sensors (array-like): X-coordinates of the sensor data.
    - y_sensors (array-like): Y-coordinates of the sensor data.
    - z_sensors (array-like): Sensor values corresponding to (x_sensors, y_sensors).
    - x_mesh (array-like): X-coordinates for which to interpolate.
    - y_mesh (array-like): Y-coordinates for which to interpolate.
    - method (str, optional): Interpolation method, default is 'cubic'.

    Returns:
    - z_interpolated (ndarray): Interpolated values at the specified coordinates (x_new, y_new).
    - max_z (float): Global maximum value of the interpolated data.
    - max_indices (tuple): Indices of the global maximum in the interpolated data.

    Note:
    - The function uses scipy's griddata for 2D interpolation.
    - The default interpolation method is cubic, but other methods can be specified.
    """
    # Performing 2D interpolation
    z_interpolated = griddata((self.x_sensors, self.y_sensors), self.z_sensors, (self.x_mesh, self.y_mesh), method=method)

    # Finding Global Max and its indices
    max_z = np.nanmax(z_interpolated)
    max_indices = np.where(z_interpolated == max_z)

    return z_interpolated, max_z, max_indices
  
 def gaussian_process_interpolation(self, length_scale=10):
    """
    Perform Gaussian Process interpolation on the given data.

    Parameters:
    - X_train (array-like): Training data features.
    - y_train (array-like): Training data labels.
    - x_new (array-like): New data features for interpolation.
    - y_new (array-like): New data labels for interpolation.
    - length_scale (float, optional): Length scale for the RBF kernel. Default is 1.

    Returns:
    - reshaped_gp_results (numpy.ndarray): Interpolated values reshaped to the shape of x_new and y_new.
    - max_z_gp (float): Global maximum value in the interpolated grid.
    - max_indices_gp (tuple): Indices of the global maximum value in the interpolated grid.
    """

    X_train = np.column_stack((self.x_sensors, self.y_sensors))
    y_train = self.z_sensors

    # Define the kernel for Gaussian Process (RBF kernel is used here)
    kernel = RBF(length_scale=length_scale)

    # Create the Gaussian Process Regressor
    gp_model = GaussianProcessRegressor(kernel=kernel)

    # Fit the model with the data
    gp_model.fit(X_train, y_train)

    # Make some test data
    X_test = np.column_stack((self.x_mesh.ravel(), self.y_mesh.ravel()))

    # Predict interpolated values
    interpolated_values_gp, std = gp_model.predict(X_test, return_std=True)
    reshaped_gp_results = interpolated_values_gp.reshape(self.x_mesh.shape)

    # Finding Global Max and its indices using the Gaussian process model
    max_z_gp = np.nanmax(reshaped_gp_results)
    max_indices_gp = np.where(max_z_gp == reshaped_gp_results)

    return reshaped_gp_results, max_z_gp, max_indices_gp

def random_forest_interpolation(self, max_depth=2, n_estimators=50, random_state=42):
    """
    Perform Random Forest interpolation on the given data.

    Parameters:
    - X_train (array-like): Training data features.
    - y_train (array-like): Training data labels.
    - x_new (array-like): New data features for interpolation.
    - y_new (array-like): New data labels for interpolation.
    - max_depth (int, optional): Maximum depth of the decision trees. Default is 2.
    - n_estimators (int, optional): Number of trees in the forest. Default is 50.
    - random_state (int, optional): Seed for random number generation. Default is 42.

    Returns:
    - reshaped_rf_results (numpy.ndarray): Interpolated values reshaped to the shape of x_new and y_new.
    - max_z_rf (float): Global maximum value in the interpolated grid.
    - max_indices_rf (tuple): Indices of the global maximum value in the interpolated grid.
    """

    X_train = np.column_stack((self.x_sensors, self.y_sensors))
    y_train = self.z_sensors

    # Create the Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    # Fit the model with the data
    rf_model.fit(X_train, y_train)

    # Make some test data
    X_test = np.column_stack((self.x_mesh.ravel(), self.y_mesh.ravel()))

    # Predict interpolated values
    interpolated_values_rf = rf_model.predict(X_test)
    reshaped_rf_results = interpolated_values_rf.reshape(self.x_mesh.shape)

    # Finding Global Max and its indices using the Random Forest model
    max_z_rf = np.nanmax(reshaped_rf_results)
    max_indices_rf = np.unravel_index(np.argmax(reshaped_rf_results), reshaped_rf_results.shape)

    return reshaped_rf_results, max_z_rf, max_indices_rf
