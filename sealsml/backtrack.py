# Functions for B Travis 
import numpy as np
from typing import Tuple, List
import math

def pathmax(factor_x=[0.4], x_width, factor_y=[0.4], y_width):
    """
    This function calculates the pathmax, which is the minimum of the product of factor_x and x_width and the product of factor_y and y_width.

    x_width and y_width should be in meters
    
    Args:
        factor_x: The factor to multiply by x_width. Should be a NumPy array with the same shape as x_width and y_width.
        x_width: The width in the x-direction. Should be a NumPy array with the same shape as factor_x and y_width.
        factor_y: The factor to multiply by y_width. Should be a NumPy array with the same shape as factor_x and y_width.
        y_width: The width in the y-direction. Should be a NumPy array with the same shape as factor_x and y_width.

    Returns:
        The pathmax value as a NumPy array.

    Raises:
        TypeError: If any of the inputs are not NumPy arrays or do not have the same shape.
        ValueError: If any of the factors are not within the valid range [0, 0.75].
    """
    # Convert all inputs to NumPy arrays
    factor_x = np.asarray(factor_x)
    x_width = np.asarray(x_width)
    factor_y = np.asarray(factor_y)
    y_width = np.asarray(y_width)

    # Check if all inputs have the same shape
    if not (factor_x.shape == x_width.shape == factor_y.shape == y_width.shape):
        raise TypeError("All inputs must have the same shape.")

    # Check if factors are within the valid range [0, 0.75]
    if not (0 <= np.min(factor_x) <= 0.75) or not (0 <= np.min(factor_y) <= 0.75):
        raise ValueError("Factors should range between 0 and 0.75.")

    # Calculate pathmax using NumPy operations
    return np.minimum(factor_x * x_width, factor_y * y_width)

def findmaxCH4(CH4: np.ndarray, times: np.ndarray) -> Tuple[float, float, int]:
    """
    Finds the first occurrence of the maximum CH4 concentration in a time series for a sensor.

    Args:
        CH4 (np.ndarray): A NumPy array of CH4 concentration values.
        times (np.ndarray): A NumPy array of corresponding timestamps.

    Returns:
        tuple: A tuple containing the following:
            maxC (float): The first occurrence of the maximum CH4 concentration.
            time_maxC (float): The time at which the first maximum CH4 concentration occurs.
            ijkmax (int): The index in the time series for the first maximum CH4 concentration.

    Raises:
        ValueError: If the lengths of the CH4 and times arrays are not equal.

    Notes:
        * If no maximum is found (e.g., all values are constant or zero), the function
          returns the value and time at the midpoint of the time series.
        * This is an arbitrary choice, and other strategies could be used (e.g.,
          returning NaN or None).
    """
    if not CH4.shape == times.shape:
        raise ValueError("The shapes of the CH4 and times arrays must be equal.")

    # Find the first occurrence of the maximum value and its index
    max_idx = np.argmax(CH4)
    max_c = CH4[max_idx]
    time_max_c = times[max_idx]

    # Use conditional logic to handle cases where no maximum was found
    if max_c == 0.0 or max_idx == 0:
        # Use midpoint in case of constant or zero values
        max_idx = CH4.shape[0] // 2
        max_c = CH4[max_idx]
        time_max_c = times[max_idx]

    return max_c, time_max_c, max_idx

def backtrack(ijk_start: int, u_sonic, v_sonic, dt, sensor_x, sensor_y, pathmax):
    """
    Backtracks along a velocity path until a specified distance is traversed and returns the average velocity vector.

    Args:
        ijk_start (int): Index in the time series at which to start backtracking. (time step)
        u_sonic (list): List of x-component wind values at the sonic anemometer vs time. (m/s)
        v_sonic (list): List of y-component wind values at the sonic anemometer vs time. (m/s)
        dt (float): Time step size. 
        sensor_x (float): X-coordinate of the sensor. 
        sensor_y (float): Y-coordinate of the sensor.
        pathmax (float): Maximum backtrack path length. (distance in meters?)

    Returns:
        Scaled U and V wind componets. These are post-processed later. 

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

    if not all(np.size(arg) == 1 for arg in [ijk_start, sensor_x, sensor_y, pathmax]):
        raise ValueError("ijk_start, sensor_x, sensor_y, and pathmax should all have a length of 1.")

    # Initialize variables
    xn = sensor_x
    yn = sensor_y
    ijk = ijk_start
    ux_sum = 0.0
    vy_sum = 0.0
    dx = 0.0
    dy = 0.0
    total_dist = 0.0
    HALF = 0.5

    # Backtrack along the velocity path
    while total_dist < pathmax and ijk > 0:
        u_current, u_previous = u_sonic[ijk], u_sonic[ijk - 1]
        v_current, v_previous = v_sonic[ijk], v_sonic[ijk - 1]

        u_bar = HALF * (u_current + u_previous)
        v_bar = HALF * (v_current + v_previous)

        xnm1 = xn - dt * u_bar
        ynm1 = yn - dt * v_bar
        ijk -= 1
        ux_sum += u_bar
        vy_sum += v_bar
        xn = xnm1
        yn = ynm1
        # Calculating Distance
        dx = sensor_x - xn
        dy = sensor_y - yn
        total_dist = np.sqrt((dx**2 + dy**2))

    # Compute average horizontal wind components
    denominator = max(1, (ijk_start - ijk))
    
    avg_u = ux_sum / denominator
    avg_v = vy_sum / denominator

    return avg_u, avg_v