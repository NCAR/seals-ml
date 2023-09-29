import numpy as np

class geo:
  def __init__(self, reference_array, target_array):
    self.reference_array = reference_array
    self.target_array = target_array
  
  def distance_between_points_3d(self, grid_resolution=2):
    """Calculate the distance between two points in 3D space (x, y, and z).
    Args:
      reference_array: The coordinates of the first point. A NumPy array of shape (3,) or (n, 3).
      target_array: The coordinates of the second point. A NumPy array of shape (3,) or (n, 3).
      grid_resolution: the scaler to go from i, j, k to real world distance. Assumes that it is the same for i, j, k
    Returns:
      A NumPy array of distances.

    Raises:
          TypeError: If either `reference_array` or `target_array` is not a NumPy array, or if either
            array does not have shape (3,) or (n, 3).
    """
    # Check that the reference_array and target_array arguments are NumPy arrays
    if not isinstance(self.reference_array, np.ndarray):
      raise TypeError("The `reference_array` argument must be a NumPy array.")

    if not isinstance(self.target_array, np.ndarray):
      raise TypeError("The `target_array` argument must be a NumPy array.")

    # Check they have the correct shape, needs to be 3 columns
    if self.reference_array.shape != (3,) and self.reference_array.shape[1] != 3:
      raise TypeError("reference_array must have shape (3,) or (x, 3).")

    if self.target_array.shape != (3,) and self.target_array.shape[1] != 3:
      raise TypeError("target_array must have shape (3,) or (x, 3).")

    # Calculate the distance between the two points.
    distance = np.linalg.norm(self.reference_array - self.target_array, axis=1)
    true_distance = grid_resolution * distance

    return true_distance

  def calculate_azimuth(self):
    """Calculates the azimuth between two points.

    Args:
      reference_array: A NumPy array of shape (n, 3) representing the first point.
      target_array: A NumPy array of shape (n, 3) representing the second point.

    Returns:
      The azimuth in degrees, from 0 to 360, in a NumPy array of the same shape as the input arrays.
    """
    # Check they have the correct shape, needs to be 3 columns
    if self.reference_array.shape != (3,) and self.reference_array.shape[1] != 3:
      raise TypeError("reference_array must have shape (3,) or (x, 3).")

    if self.target_array.shape != (3,) and self.target_array.shape[1] != 3:
      raise TypeError("target_array must have shape (3,) or (x, 3).")

    # Extract the x, y, and z coordinates from each array.
    if len(self.reference_array.shape) == 1:
      x1, y1 = self.reference_array[0], self.reference_array[1]
    else:
      x1, y1 = self.reference_array[:, 0], self.reference_array[:, 1]

    if len(self.target_array.shape) == 1:
      x2, y2 = self.target_array[0], self.target_array[1]
    else:
      x2, y2 = self.target_array[:, 0], self.target_array[:, 1]

    # Calculate the difference between the x-coordinates and y-coordinates
    xdif = x2 - x1
    ydif = y2 - y1

    if (xdif == 0).any() and (ydif == 0).any():
      print('Warning, Azimuth Calculation might be wrong for vertically offset points')

    # Calculate the azimuth for the current pair of points
    azi_rad = np.arctan2(ydif, xdif)
    azi_deg = -1 * np.degrees(azi_rad)
    azimuth_deg = (azi_deg + 90 + 360) % 360

    return azimuth_deg

  def calculate_elevation_angle(self):
    """Calculates the elevation angle between two points/arrays.

      Commonly called the dip if you are a geologist :)

    Args:
      reference_array: A NumPy array of shape (n, 3) or (3,) representing the first point.
      target_array: A NumPy array of shape (n, 3) representing the second array.

    Returns:
      The elevation angle in degrees, from 0 to 90.

    Raises:
      TypeError: If either `reference_array` or `target_array` is not a NumPy array of shape (n, 3).

      ValueError: If the two points do not have the same shape.
    """

        # Check that the reference_array and target_array arguments are NumPy arrays
    if not isinstance(self.reference_array, np.ndarray):
      raise TypeError("The `reference_array` argument must be a NumPy array.")

    if not isinstance(self.target_array, np.ndarray):
      raise TypeError("The `target_array` argument must be a NumPy array.")

      # Check they have the correct shape, needs to be 3 columns
    if self.reference_array.shape != (3,) and self.reference_array.shape[1] != 3:
      raise TypeError("reference_array must have shape (3,) or (x, 3).")

    if self.target_array.shape != (3,) and self.target_array.shape[1] != 3:
      raise TypeError("target_array must have shape (x, 3).")

      ## Calculate dip
    if len(self.reference_array.shape) == 1 and len(self.target_array.shape) == 1:
      dz = self.target_array[2] - self.reference_array[2]
      distance = np.linalg.norm(self.target_array - self.reference_array)
    elif len(self.reference_array.shape) == 1 and len(self.target_array.shape) == 2:
      dz = self.target_array[:, 2] - self.reference_array[2]
      distance = np.linalg.norm(self.target_array - self.reference_array, axis=1)
    else:
      dz = self.target_array[:, 2] - self.reference_array[:, 2]
      distance = np.linalg.norm(self.target_array - self.reference_array, axis=1)

    dip_radians = np.arcsin(dz / distance)
    # let's convert to degrees and round it
    elevation_angle = np.degrees(dip_radians)

    return elevation_angle