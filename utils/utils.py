import xarray
import numpy as np

def distance_between_points_3d(ds, point1, point2):
  """
  Calculate the distance between two points in a xarray dataset in x, y, and z.

  Args:
    ds: The xarray dataset.
    point1: The coordinates of the first point. A tuple of three numbers.
    point2: The coordinates of the second point. A tuple of three numbers.

  Returns:
    The distance between the two points.

  Raises:
    TypeError: If either `point1` or `point2` is not a tuple of three numbers.
    PointOutOfBoundsError: If either point is not within the xarray dataset.
  """

  # Check that the point1 and point2 arguments are tuples of three numbers.
  if not isinstance(point1, tuple) or len(point1) != 3:
    raise TypeError("The `point1` argument must be a tuple of three numbers.")

  if not isinstance(point2, tuple) or len(point2) != 3:
    raise TypeError("The `point2` argument must be a tuple of three numbers.")

  # Check that the points are within the xarray dataset.
  for point in (point1, point2):
    if point[0] > ds.x.values.max():
      raise ValueError("The `point1` x-coordinate is out of bounds.")
    if point[1] > ds.y.values.max():
      raise ValueError("The `point1` y-coordinate is out of bounds.")
    if point[2] > ds.z.values.max():
      raise ValueError("The `point1` z-coordinate is out of bounds.")

  # Calculate the distance between the two points.
  distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)

  return distance