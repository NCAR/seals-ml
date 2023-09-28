import numpy as np

class geo:
  def __init__(self, array1, array2):
    self.array1 = array1
    self.array2 = array2

  def distance_between_points_3d(self, grid_resolution=2):
    """
    Calculate the distance between two points in 3D space (x, y and z).

    Args:
      array: The coordinates of the first point. A NumPy array of shape (3,) or (n, 3).
      array1: The coordinates of the second point. A NumPy array of shape (3,) or (n, 3).
      grid_resoultion: the scaler to go from i,j,k to real world distance. Assumes that it is the same for i,j,k
    Returns:
      A NumPy array of distances.

    Raises:
      TypeError: If either `array` or `array1` is not a NumPy array, or if either
        array does not have shape (3,) or (n, 3).
    """

    # Check that the array and array1 arguments are NumPy arrays
    if not isinstance(self.array1, np.ndarray):
      raise TypeError("The `array` argument must be a NumPy array.")

    if not isinstance(self.array2, np.ndarray):
      raise TypeError("The `array1` argument must be a NumPy array.")

    # Check they have the correct shape, needs to be 3 columns
    if self.array1.shape != (3,) and self.array1.shape[1] != 3:
      raise TypeError("array must have shape (3,) or (x, 3).")
  
    if self.array2.shape != (3,) and self.array2.shape[1] != 3:
      raise TypeError("array must have shape (3,) or (x, 3).")
   
    # Calculate the distance between the two points.
    distance = np.linalg.norm(self.array1 - self.array2, axis=1)
    true_distance = grid_resolution*distance
    
    return true_distance

  def calculate_azimuth(self):
    """Calculates the azimuth between two points.

    Args:
      array1: A NumPy array of shape (n, 3) representing the first point.
      array2: A NumPy array of shape (n, 3) representing the second point.

    Returns:
      The azimuth in degrees, from 0 to 360, in a NumPy array of the same shape as the input arrays.
    """
    # Check they have the correct shape, needs to be 3 columns
    if self.array1.shape != (3,) and self.array1.shape[1] != 3:
      raise TypeError("array must have shape (3,) or (x, 3).")

    if self.array2.shape != (3,) and self.array2.shape[1] != 3:
      raise TypeError("array must have shape (3,) or (x, 3).")

    # Extract the x, y, and z coordinates from each array.
    if len(self.array1.shape) == 1:
      x1, y1 = self.array1[0], self.array1[1]
    else:
      x1, y1 = self.array1[:, 0], self.array1[:, 1]
    
    if len(self.array2.shape) == 1:
      x2, y2 = self.array2[0], self.array2[1]
    else:
      x2, y2 = self.array2[:, 0], self.array2[:, 1]

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

  def calculate_elevation_angle(self):
    """Calculates the elevation angle between two points/arrays.
  
    Commonly called the dip if you are a geologist :) 
  
    Args:
      array1: A NumPy array of shape (n, 3) or (3,) representing the first point.
      array2: A NumPy array of shape (n, 3) representing the second array.

    Returns:
      The elevation angle in degrees, from 0 to 90.

    Raises:
      TypeError: If either `array` or `array1` is not a NumPy array of shape (n, 3).

      ValueError: If the two points do not have the same shape.
    """

    # Check that the array and array1 arguments are NumPy arrays
    if not isinstance(self.array1, np.ndarray):
      raise TypeError("The `array` argument must be a NumPy array.")

    if not isinstance(self.array2, np.ndarray):
      raise TypeError("The `array1` argument must be a NumPy array.")

    # Check they have the correct shape, needs to be 3 columns
    if self.array1.shape != (3,) and self.array1.shape[1] != 3:
      raise TypeError("array must have shape (3,) or (x, 3).")
  
    if self.array2.shape != (3,) and self.array2.shape[1] != 3:
      raise TypeError("array2 must have shape (x, 3).")

    ## Calculate dip
    if len(self.array1.shape) == 1 and len(self.array2.shape) == 1:
      dz = self.array2[2] - self.array1[2]
      distance = np.linalg.norm(self.array2 - self.array1)
    elif len(self.array1.shape) == 1 and len(self.array2.shape) == 2:
      dz = self.array2[:,2] - self.array1[2]
      distance = np.linalg.norm(self.array2 - self.array1, axis=1)
    else:
      dz = self.array2[:,2] - self.array1[:,2]
      distance = np.linalg.norm(self.array2 - self.array1, axis=1)
                                       
    dip_radians = np.arctan2(dz, distance)
    # let's convert to degrees and round it
    elevation_angle = np.degrees(dip_radians)

    return elevation_angle
