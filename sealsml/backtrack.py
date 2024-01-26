import numpy as np
import pandas as pd
import math

def polar(u, v):
    """
    Converts u,v time series at a point to wind speed and wind direction using only numpy.

    Args:
        u (numpy.ndarray): 1D array of horizontal wind components in the x-direction (East-West).
        v (numpy.ndarray): 1D array of horizontal wind components in the y-direction (North-South).

    Returns:
        tuple: A tuple containing the following:
            angle (numpy.ndarray): 1D array of wind directions in degrees (0 = East, counterclockwise).
            speed (numpy.ndarray): 1D array of wind speeds.

    Raises:
        ValueError: If u and v have different lengths.

    """

    if len(u) != len(v):
        raise ValueError("u and v must have the same length.")

    speed = np.sqrt(u**2 + v**2)

    # Use numpy's arctangent-2 function with correction for quadrant
    angle = np.rad2deg(np.arctan2(v, u))
    angle[angle < 0] += 360

    return angle, speed

def polar2d(u2d, v2d, m, n):
    """
    Computes wind speed and direction for a 2D u,v wind field.

    Args:
        u2d (numpy.ndarray): 2D array of horizontal wind components in the x-direction.
        v2d (numpy.ndarray): 2D array of horizontal wind components in the y-direction.
        m (int): Number of rows in the u2d and v2d arrays.
        n (int): Number of columns in the u2d and v2d arrays.

    Returns:
        tuple: A tuple containing the following:
            angle2d (numpy.ndarray): 2D array of wind directions in degrees (0 = East, counterclockwise).
            speed2d (numpy.ndarray): 2D array of wind speeds.

    Raises:
        ValueError: If u2d and v2d have different shapes.

    Notes:
        * The function assumes that u2d and v2d have the same dimensions.
        * Wind directions are calculated using the atan2 function, which accounts for quadrant differences.

    """

    if u2d.shape != v2d.shape:
        raise ValueError("u2d and v2d must have the same shape.")

    angle2d = np.zeros_like(u2d)
    speed2d = np.zeros_like(u2d)

    for i in range(m):
        for j in range(n):
            speed2d[i][j] = np.sqrt(u2d[i][j]**2 + v2d[i][j]**2)
            angle2d[i][j] = math.degrees(math.atan2(v2d[i][j], u2d[i][j]))

    return angle2d, speed2d

def findmaxC(CH4, times, n):
    """
    Finds the maximum CH4 concentration in a time series for a sensor.

    Args:
        CH4 (list): A list of CH4 concentration values.
        times (list): A list of corresponding timestamps.
        n (int): The length of the CH4 and times lists.

    Returns:
        tuple: A tuple containing the following:
            maxC (float): The maximum CH4 concentration.
            time_maxC (float): The time at which the maximum CH4 concentration occurs.
            ijkmax (int): The index in the time series for the maximum CH4 concentration.

    Raises:
        ValueError: If the lengths of the CH4 and times lists are not equal.

    Notes:
        * If no maximum is found (e.g., all values are constant or zero), the function
          returns the value and time at the midpoint of the time series.
        * This is an arbitrary choice, and other strategies could be used (e.g.,
          returning NaN or None).

    """

    if len(CH4) != len(times):
        raise ValueError("The lengths of the CH4 and times lists must be equal.")

    max_c = 0.0
    time_max_c = 0.0
    ijk_max = 0

    for i in range(n):
        if CH4[i] > max_c:
            max_c = CH4[i]
            time_max_c = times[i]
            ijk_max = i

    # If no maximum is found, return the midpoint value

    if max_c == 0.0 or ijk_max == 0:
        ijk_max = int(n / 2)
        time_max_c = times[ijk_max]

    return max_c, time_max_c, ijk_max

def backtrack(ijk_start, u_sonic, v_sonic, dt, sensor_x, sensor_y, pathmax):
    """
    Backtracks along a velocity path until a specified distance is traversed and returns the average velocity vector.

    Args:
        ijk_start (int): Index in the time series at which to start backtracking.
        u_sonic (list): List of x-component wind values at the sonic anemometer vs time.
        v_sonic (list): List of y-component wind values at the sonic anemometer vs time.
        dt (float): Time step size.
        sensor_x (float): X-coordinate of the sensor.
        sensor_y (float): Y-coordinate of the sensor.
        pathmax (float): Maximum backtrack path length.

    Returns:
        tuple: A tuple containing the following:
            avg_u (float): Average x-component wind vector component over the backtrack time interval.
            avg_v (float): Average y-component wind vector component over the backtrack time interval.

    Raises:
        ValueError: If the length of u_sonic and v_sonic lists are not equal.

    Notes:
        * The function assumes that u_sonic and v_sonic have the same length.
        * The function stops backtracking if it reaches the beginning of the time series (ijk = 0) or if the total distance traveled exceeds pathmax.

    """

    if len(u_sonic) != len(v_sonic):
        raise ValueError("The lengths of u_sonic and v_sonic lists must be equal.")

    # Initialize variables
    xn = sensor_x
    yn = sensor_y
    ijk = ijk_start
    ux_sum = 0.0
    vy_sum = 0.0
    xdist = 0.0
    ydist = 0.0
    total_dist = 0.0

    # Backtrack along the velocity path
    while (total_dist < pathmax) and ijk > 0:
        u_bar = 0.5 * (u_sonic[ijk] + u_sonic[ijk - 1])
        v_bar = 0.5 * (v_sonic[ijk] + v_sonic[ijk - 1])
        xnm1 = xn - dt * u_bar
        ynm1 = yn - dt * v_bar
        ijk -= 1
        ux_sum += u_bar
        vy_sum += v_bar
        xn = xnm1
        yn = ynm1
        xdist = abs(sensor_x - xn)
        ydist = abs(sensor_y - yn)
        total_dist = np.sqrt(xdist**2 + ydist**2)

    # Compute average horizontal wind components
    avg_u = ux_sum / max(1, (ijk_start - ijk))
    avg_v = vy_sum / max(1, (ijk_start - ijk))

    return avg_u, avg_v
