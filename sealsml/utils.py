import xarray as xr
import numpy as np


def points_within_dataset(ds, point1, point2):
  """
  check that two points (x, y, and z) are in a xarray dataset.

  Args:
    ds: The xarray dataset.
    point1: The coordinates of the first point. A tuple of three numbers.
    point2: The coordinates of the second point. A tuple of three numbers.

  Returns:
    print statement

  Raises:
    TypeError: If either `point1` or `point2` is not a tuple of three numbers.
    ValueError: If either point is not within the xarray dataset.
  """

  # Check that the point1 and point2 arguments are tuples of three numbers.
  if not isinstance(point1, tuple) or len(point1) != 3:
    raise TypeError("The `point1` argument must be a tuple of three numbers.")

  if not isinstance(point2, tuple) or len(point2) != 3:
    raise TypeError("The `point2` argument must be a tuple of three numbers.")

  # Check that the points are within the xarray dataset.
  for point in (point1, point2):
    if np.any(np.isin(ds.x.values, point[0])) != True:
      raise ValueError("The x-coordinate is out of bounds.")
    if np.any(np.isin(ds.y.values, point[1])) != True:
      raise ValueError("The y-coordinate is out of bounds.")
    if np.any(np.isin(ds.z.values, point[2])) != True:
      raise ValueError("The z-coordinate is out of bounds.")
  
  print('both points are within xarray dataset')

def distance_between_points_3d(point1, point2):
  """
  Calculate the distance between two points in 3D space (x, y and z).

  Args:
    point1: The coordinates of the first point. A tuple of three numbers.
    point2: The coordinates of the second point. A tuple of three numbers.

  Returns:
    The distance between the two points.

  Raises:
    TypeError: If either `point1` or `point2` is not a tuple of three numbers.
  """

  # Check that the point1 and point2 arguments are tuples of three numbers.
  if not isinstance(point1, tuple) or len(point1) != 3:
    raise TypeError("The `point1` argument must be a tuple of three numbers.")

  if not isinstance(point2, tuple) or len(point2) != 3:
    raise TypeError("The `point2` argument must be a tuple of three numbers.")

  # Calculate the distance between the two points.
  distance = np.linalg.norm(np.array(point2) - np.array(point1))

  return distance

def azimuth(point1, point2):
  """Calculates the azimuth between two points.

  Args:
    point1: A tuple of three floats representing the first point.
    point2: A tuple of three floats representing the second point.

  Returns:
    The azimuth in degrees, from 0 to 360.
  """
  # Check that the point1 and point2 arguments are tuples of three numbers.
  if not isinstance(point1, tuple) or len(point1) != 3:
    raise TypeError("The `point1` argument must be a tuple of three numbers.")

  if not isinstance(point2, tuple) or len(point2) != 3:
    raise TypeError("The `point2` argument must be a tuple of three numbers.")

  x1, y1 = point1[:2]
  x2, y2 = point2[:2]

  dlon = x2 - x1
  y = np.sin(dlon) * np.cos(y2)
  x = np.cos(y1) * np.sin(y2) - np.sin(y1) * np.cos(y2) * np.cos(dlon)
  theta = np.arctan2(y, x)
  azimuth = np.round((theta * 180 / np.pi + 360), 2)
  return azimuth

def dip(point1, point2):
    """Calculates the dip between two points.

    Args:
        point1: A tuple of three floats representing the first point.
        point2: A tuple of three floats representing the second point.

    Returns:
        The dip in degrees, from 0 to 90.
    """
    # Check that the point1 and point2 arguments are tuples of three numbers.
    if not isinstance(point1, tuple) or len(point1) != 3:
        raise TypeError("The `point1` argument must be a tuple of three numbers.")

    if not isinstance(point2, tuple) or len(point2) != 3:
        raise TypeError("The `point2` argument must be a tuple of three numbers.")

    # Calculate the distance between the two points.
    distance = np.round(np.linalg.norm(np.array(point2) - np.array(point1)),2)
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    dz = point2[2] - point1[2]
    dip = np.round(np.arctan2(dz, distance) * 180 / np.pi, 2)
    return dip


class DataSampler(object):

    def __init__(self, min_trace_sensors=3, max_trace_sensors=15, n_leaks=1, reference_loc=(15, 15),
                 sensor_mask_radius=50, sensor_height=3, leak_radius=10, emission_var='q_CH4'):

        self.min_trace_sensors = min_trace_sensors
        self.max_trace_sensors = max_trace_sensors
        self.n_leaks = n_leaks
        self.reference_loc = reference_loc
        self.sensor_mask_radius = sensor_mask_radius
        self.sensor_height = sensor_height
        self.leak_radius = leak_radius
        self.emission_var = emission_var

    def load_data(self, file_names):

        self.data = xr.open_mfdataset(file_names, parallel=True)
        self.data = self.data.assign_coords({'iDim': self.data['iDim'],
                                             'jDim': self.data['jDim'],
                                             'kDim': self.data['kDim']})

        self.time_steps = len(self.data['timeDim'].values)

    def sample(self, window_size, samples_per_window):

        for t in np.arange(0, self.time_steps - window_size):

            n_sensors = np.random.randint(low=self.min_trace_sensors,
                                          high=self.max_trace_sensors,
                                          size=samples_per_window)

            i = np.random.randint(low=0, high=30, size=n_sensors)
            j = np.random.randint(low=0, high=30, size=n_sensors)

            sensor_sample = self.data[['u', 'v', 'w', self.emission_var]].isel(timeDim=slice(t, t + window_size),
                                                                               kDim=self.sensor_height,
                                                                               jDim=j,
                                                                               iDim=i).to_array()






