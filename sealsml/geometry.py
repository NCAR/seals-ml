import numpy as np
import pandas as pd


class GeoCalculator(object):
  def __init__(self, ref_array=None, target_array=None):

      self.reference_array = ref_array
      self.target_array = target_array

  def get_geometry(self, ref_array, target_array, grid_resolution=False, pd_export=False,
              column_names=("distance", "azimuth_cos", "azimuth_sin", "elevation_angle")):
    """
    Calculates the geometric metrics between two arrays of points.

    Args:
        ref_array (numpy.ndarray): A numpy array of points, with shape (n, 3). Usually the reference points.
        target_array (numpy.ndarray): A numpy array of points, with shape (n, 3). Usually the target points.
        pd_export (bool, optional): Whether to export the results as a Pandas DataFrame. Defaults to False.
        column_names (list[str], optional): The column names for the exported Pandas DataFrame. Defaults to ["distance", "azimuth_cos", "azimuth_sin", "elevation_angle"].

    Returns:
        numpy.ndarray or pd.DataFrame: The geometric metrics, with shape (n, 4). If pd_export is True, a Pandas DataFrame is returned.
    """
    self.reference_array = ref_array
    self.target_array = target_array

    ### Let's get these metrics!
    # distance
    distance = self.distance_between_points_3d(grid_resolution=grid_resolution)

    # azimuth
    azi = self.calculate_azimuth()
    azi_cos = np.cos(np.radians(azi))
    azi_sin = np.sin(np.radians(azi))
    # elevation angle
    ele_angle = self.calculate_elevation_angle()

    combined_array = np.column_stack((distance, azi_cos, azi_sin, ele_angle))
    if pd_export:
      return pd.DataFrame(combined_array, columns=column_names)
    else:
      return combined_array

  def distance_between_points_3d(self, grid_resolution=False):
    """Calculate the distance between two points in 3D space (x, y, and z).
    Args:
      reference_array: The coordinates of the first point. A NumPy array of shape (3,) or (n, 3).
      target_array: The coordinates of the second point. A NumPy array of shape (3,) or (n, 3).
      grid_resolution: the scaler to go from i, j, k to real world distance. Assumes that it is the same for i, j, k
      Also can be false if using real coordinates in x,y,z space 
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

    if grid_resolution == False:
      # Calculate the distance between the two points.
      true_distance = np.linalg.norm(self.reference_array - self.target_array, axis=1)
  
    elif isinstance(grid_resolution, (float, int)):
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
      The elevation angle in degrees, from -90 to 90.

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
      horz_distance = np.linalg.norm(self.target_array[0:2] - self.reference_array[0:2])
    elif len(self.reference_array.shape) == 1 and len(self.target_array.shape) == 2:
      dz = self.target_array[:, 2] - self.reference_array[2]
      horz_distance = np.linalg.norm(self.target_array[:,:2] - self.reference_array[0:2], axis=1)
    else:
      dz = self.target_array[:, 2] - self.reference_array[:, 2]
      horz_distance = np.linalg.norm(self.target_array[:,:2] - self.reference_array[:,:2], axis=1)

    dip_radians = np.arctan2(dz , horz_distance)
    # let's convert to degrees and round it
    elevation_angle = np.degrees(dip_radians)

    return elevation_angle

def get_relative_azimuth(u, v, x_ref, y_ref, z_ref, x_target, y_target, z_target, time_series=True):
    """
    Calculates the relative azimuth between a reference point and a target point based on wind vectors.

    Parameters:
    - u (array-like): Wind vector component in the x-direction.
    - v (array-like): Wind vector component in the y-direction.
    - x_ref (float): X-coordinate of the reference point.
    - y_ref (float): Y-coordinate of the reference point.
    - z_ref (float): Z-coordinate of the reference point.
    - x_target (array-like): X-coordinate(s) of the target point(s).
    - y_target (array-like): Y-coordinate(s) of the target point(s).
    - z_target (array-like): Z-coordinate(s) of the target point(s).
    - time_series (bool, optional): If True, returns a time series of results. Defaults to True.

    Returns:
    - array-like: If time_series is True, returns a 2D array containing positional variables and rotated wind vectors and a 
                  scalar value of the mean(in time) wind direction.
                  If time_series is False, returns a 1D array containing positional variables and rotated wind vectors 
                  for the first time step and the wind direction.
    """
    # Calculate the mean wind direction angle
    ubar = u.mean()
    vbar = v.mean()
    theta_wd = np.arctan2(vbar, ubar)
    
    # Calculate the relative coordinates of the target point with respect to the reference point
    x_relative = x_target - x_ref
    y_relative = y_target - y_ref
    
    # Rotate the relative coordinates to align with the wind direction
    x_rotated = x_relative * np.cos(-theta_wd) - y_relative * np.sin(-theta_wd)
    y_rotated = x_relative * np.sin(-theta_wd) + y_relative * np.cos(-theta_wd)
    
    # Calculate the radial distance from the reference point to the rotated target point
    radius_rotated = np.sqrt(x_rotated ** 2 + y_rotated ** 2)
    
    # Calculate the Euclidean distance between the reference and target points in three-dimensional space
    distance = np.linalg.norm(np.column_stack([x_target, y_target, z_target]) -
                              np.column_stack([x_ref, y_ref, z_ref]).flatten(), axis=1)
    
    # Calculate the azimuth angle from the reference point to the rotated target point
    theta = np.arctan2(y_rotated, x_rotated)
    
    # Calculate the elevation angle between the reference and target points
    elevation_theta = np.arctan2(z_target - z_ref, distance)
    
    # Rotate the wind vectors to align with the rotated coordinate system
    u_rot = u * np.cos(-theta_wd) - v * np.sin(-theta_wd)
    v_rot = u * np.sin(-theta_wd) + v * np.cos(-theta_wd)
   
    # Construct an array containing positional variables and rotated wind vectors
    pos_vars = np.column_stack([radius_rotated, np.sin(theta), np.cos(theta), elevation_theta])
    
    # If time_series is True, repeat positional variables and rotated wind vectors, stack them, and transpose the result
    if time_series:
        return np.column_stack([np.repeat(pos_vars, u.size, axis=0), u_rot, v_rot]).T, theta_wd
    
    # If time_series is False, stack positional variables, provide the rotated mean wind components, and transpose the result
    else:
        return np.column_stack([pos_vars, ubar * np.cos(-theta_wd) - vbar * np.sin(-theta_wd), 
                                          ubar * np.sin(-theta_wd) + vbar * np.cos(-theta_wd)]).T, theta_wd

def polar_to_cartesian(distance, ref_azi_sin, ref_azi_cos):
    """
    Convert polar coordinates to Cartesian coordinates.

    Parameters:
    - distance (float or np.array): Radial distance or array of radial distances
    - ref_azi_sin (float or np.array): Sine of the reference azimuth angle
    - ref_azi_cos (float or np.array): Cosine of the reference azimuth angle

    Returns:
    - np.array: Cartesian coordinates, each row containing [x, y]
    """
    # Convert inputs to arrays if they are not already
    distance = np.asarray(distance)
    ref_azi_sin = np.asarray(ref_azi_sin)
    ref_azi_cos = np.asarray(ref_azi_cos)
    
    # Check if the sizes of distance and ref_azi_sin/ref_azi_cos are the same
    if distance.size != ref_azi_sin.size or distance.size != ref_azi_cos.size:
        raise ValueError("The sizes of distance, ref_azi_sin, and ref_azi_cos must be the same.")
    
    # Calculate Cartesian coordinates
    x = distance * ref_azi_cos
    y = distance * ref_azi_sin
    
    return x, y

def generate_sensor_positions_min_distance(n_sensors, xVec, yVec, min_distance, placement_strategy='random', max_attempts=500):
    """
    Generate sensor positions based on a specified strategy with a minimum distance constraint.

    Parameters:
    - n_sensors (int): Number of sensor positions to generate.
    - xVec (np.ndarray): Vector of cartesian x-coordinates.
    - yVec (np.ndarray): Vector of cartesian y-coordinates.
    - min_distance (float): Minimum distance between any two sensors.
    - placement_strategy (str): Strategy to place sensors ('random', 'fenceline', 'quadrant').
    - max_attempts (int, optional): Maximum number of attempts to place a sensor before giving up. Default is 500.

    Returns:
    - i_sensor (np.ndarray): Array of i-indices of the generated sensors.
    - j_sensor (np.ndarray): Array of j-indices of the generated sensors.
    """
    def is_valid_placement(existing_points, new_point, min_distance):
        return (np.linalg.norm(existing_points - new_point, axis=1) >= min_distance).all()

    iDim = xVec.shape[0]
    jDim = yVec.shape[0]
    i_sensor = []
    j_sensor = []
    existing_points = np.zeros((n_sensors, 2))
    cnt_points = 0

    if placement_strategy == 'random':
        while cnt_points < n_sensors:
            attempts = 0
            while attempts < max_attempts:
                new_indices = (np.random.randint(0, iDim), np.random.randint(0, jDim))
                new_point = np.array([xVec[new_indices[0]], yVec[new_indices[1]]])
                if is_valid_placement(existing_points[:cnt_points], new_point, min_distance) or cnt_points == 0:
                    i_sensor.append(new_indices[0])
                    j_sensor.append(new_indices[1])
                    existing_points[cnt_points] = new_point
                    cnt_points += 1
                    break
                attempts += 1
            if attempts >= max_attempts:
                print(f"Warning: Could not place sensor {cnt_points} in {max_attempts} tries...")
                break

    elif placement_strategy == 'fenceline':
        edge_positions = [(i, 0) for i in range(iDim)] + [(i, jDim-1) for i in range(iDim)] + \
                         [(0, j) for j in range(jDim)] + [(iDim-1, j) for j in range(jDim)]
        while cnt_points < n_sensors and len(edge_positions) > 0:
            attempts = 0
            while attempts < max_attempts and len(edge_positions) > 0:
                idx = np.random.randint(len(edge_positions))
                new_indices = edge_positions.pop(idx)
                new_point = np.array([xVec[new_indices[0]], yVec[new_indices[1]]])
                if is_valid_placement(existing_points[:cnt_points], new_point, min_distance) or cnt_points == 0:
                    i_sensor.append(new_indices[0])
                    j_sensor.append(new_indices[1])
                    existing_points[cnt_points] = new_point
                    cnt_points += 1
                    break
                attempts += 1
            if attempts >= max_attempts:
                print(f"Warning: Could not place sensor {cnt_points} in {max_attempts} tries...")
                break

    elif placement_strategy == 'quadrant':
        quadrants = {
            "NW": (xVec[:iDim // 2], yVec[:jDim // 2]),
            "NE": (xVec[iDim // 2:], yVec[:jDim // 2]),
            "SW": (xVec[:iDim // 2], yVec[jDim // 2:]),
            "SE": (xVec[iDim // 2:], yVec[jDim // 2:])
        }
        sensors_per_quadrant = n_sensors // 4
        extra_sensors = n_sensors % 4

        for quadrant, (x_range, y_range) in quadrants.items():
            num_sensors = sensors_per_quadrant
            if extra_sensors > 0:
                num_sensors += 1
                extra_sensors -= 1

            for _ in range(num_sensors):
                attempts = 0
                while attempts < max_attempts:
                    x_pos = np.random.uniform(np.min(x_range), np.max(x_range))
                    y_pos = np.random.uniform(np.min(y_range), np.max(y_range))
                    new_point = np.array([x_pos, y_pos])
                    if is_valid_placement(existing_points[:cnt_points], new_point, min_distance) or cnt_points == 0:
                        i_sensor.append(np.argmin(np.abs(xVec - x_pos)))
                        j_sensor.append(np.argmin(np.abs(yVec - y_pos)))
                        existing_points[cnt_points] = new_point
                        cnt_points += 1
                        break
                    attempts += 1
                if attempts >= max_attempts:
                    print(f"Warning: Could not place sensor in {quadrant} within {max_attempts} attempts.")
                    break

    else:
        print("Invalid placement strategy. Choose from 'random', 'fenceline', or 'quadrants'.")

    if cnt_points < n_sensors:
        print(f"Warning: Only {cnt_points} sensors placed out of {n_sensors} requested.")

    return np.array(i_sensor), np.array(j_sensor)
