import xarray
import numpy as np


def points_within_dataset(ds, point1, point2):
  """
  check that two points (x, y, and z) are in a xarray dataset.

  Args:
    ds: The xarray dataset.
    point1: The coordinates of the first point. A NumPy array of shape (3,).
    point2: The coordinates of the second point. A NumPy array of shape (3,).

  Returns:
    print statement

  Raises:
    TypeError: If either `point1` or `point2` is not a NumPy array of shape (3,).
    ValueError: If either point is not within the xarray dataset.
  """

  # Check that the point1 and point2 arguments are NumPy arrays of shape (,3).
  if not isinstance(point1, np.ndarray) or point1.shape[1] != 3:
    raise TypeError("The `point1` argument must be a NumPy array of shape (3,).")

  if not isinstance(point2, np.ndarray) or point2.shape[1] != 3:
    raise TypeError("The `point2` argument must be a NumPy array of shape (3,).")

  # Check that the points are within the xarray dataset.
  is_in_x = np.isin(point1[0], ds.x.values)
  is_in_y = np.isin(point1[1], ds.y.values)
  is_in_z = np.isin(point1[2], ds.z.values)

  #if any([is_in_x, is_in_y, is_in_z]):
  #  raise ValueError("The point is not within the xarray dataset.")
  
def distance_between_points_3d(array, array1):
  """
  Calculate the distance between two points in 3D space (x, y and z).

  Args:
    array: The coordinates of the first point. A NumPy array of shape (n, 3).
    array1: The coordinates of the second point. A NumPy array of shape (n, 3).

  Returns:
    A NumPy array of distances.

  Raises:
    TypeError: If either `array` or `array1` is not a NumPy array of shape (n,3).
  """

  # Check that the array and array1 arguments are NumPy arrays of shape (n,3).
  if not isinstance(array, np.ndarray) or array.shape[1] != 3:
    raise TypeError("The `array` argument must be a NumPy array of shape (n,3).")

  if not isinstance(array1, np.ndarray) or array1.shape[1] != 3:
    raise TypeError("The `array1` argument must be a NumPy array of shape (n,3).")

  # Calculate the distance between the two points.
  distances = np.linalg.norm(array - array1, axis=1)
  return distances

def calculate_azimuth(array, array1):
  """Calculates the azimuth between two points.

  Args:
    array: A NumPy array of shape (n, 3) representing the first point.
    array1: A NumPy array of shape (n, 3) representing the second point.

  Returns:
    The azimuth in degrees, from 0 to 360, in a NumPy array of the same shape as the input arrays.
  """

  # Check that the two arrays have the same shape.
  if not array.shape == array1.shape:
    raise ValueError("The two points must have the same shape.")

  # Extract the x, y, and z coordinates from each array.
  if len(array.shape) == 1:
    x1, y1 = array[0], array[1]
    x2, y2 = array1[0], array1[1]
  else:
    x1, y1 = array[:, 0], array[:, 1]
    x2, y2 = array1[:, 0], array1[:, 1]
  # Calculate the difference between the x-coordinates.
  xdif = x2 - x1

  # Calculate the y-component of the unit vector pointing from the first point to the second point.
  y = np.sin(xdif) * np.cos(y2)

  # Calculate the x-component of the unit vector pointing from the first point to the second point.
  x = np.cos(y1) * np.sin(y2) - np.sin(y1) * np.cos(y2) * np.cos(xdif)

  # Calculate the azimuth.
  theta = np.arctan2(y, x)
  azimuth = (theta * 180 / np.pi + 360) % 360
  azi = np.round(azimuth, 3)
  return azi

def dip(array, array1):
  """Calculates the dip between two points.

  Args:
    array: A NumPy array of shape (n, 3) representing the first point.
    array1: A NumPy array of shape (n, 3) representing the second point.

  Returns:
    The dip in degrees, from 0 to 90.

  Raises:
    TypeError: If either `array` or `array1` is not a NumPy array of shape (n, 3).

    ValueError: If the two points do not have the same shape.
  """

  # Check that the point1 and point2 arguments are NumPy arrays of shape (n, 3).
  if not isinstance(array, np.ndarray):
    raise TypeError("The `array` argument must be a NumPy array of shape (n, 3).")

  if not isinstance(array1, np.ndarray):
    raise TypeError("The `array1` argument must be a NumPy array of shape (n, 3).")

  # Check that the two points have the same shape.
  if array.shape != array1.shape:
    raise ValueError("The two points must have the same shape.")

  if len(array.shape) == 1:
    distance = np.round(np.linalg.norm(array1 - array),2)
    dz = array1[2] - array[2]
  else:
    distance = np.round(np.linalg.norm(array1 - array, axis=1), 2)
    dz = array1[:,2] - array[:,2]

  dip = np.arctan2(dz, distance) * 180 / np.pi

  # Round the dip to 2 decimal places.
  dip = np.round(dip, 2)

  return dip
