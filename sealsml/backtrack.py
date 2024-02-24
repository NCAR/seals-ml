# Functions for B Travis 
import numpy as np
from typing import Tuple, List
import math

from sealsml.geometry import GeoCalculator, polar_to_cartesian
from sealsml.baseline import remove_all_rows_with_val

def pathmax(x_width, y_width, factor_x=0.4, factor_y=0.4):
    """
    This function calculates the pathmax, which is the minimum of the product of factor_x and x_width and the product of factor_y and y_width.

    x_width and y_width should be in meters
    
    Args:
        x_width: The width in the x-direction. Should be a NumPy array with the same shape as factor_x and y_width.
        y_width: The width in the y-direction. Should be a NumPy array with the same shape as factor_x and y_width.
        factor_x: The factor to multiply by x_width. Should be a NumPy array with the same shape as x_width and y_width.
        factor_y: The factor to multiply by y_width. Should be a NumPy array with the same shape as factor_x and y_width.

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

def create_backtrack_mlp_training_data(x, num_met_sensors=1, num_sensors=3, factor_x=0.4, x_width=40,
                                       factor_y=0.4,  y_width=40, dt=1):
    """
    This function uses numpy arrays as input
    The variable should have a length of 8: ['ref_distance', 'ref_azi_sin', 'ref_azi_cos', 'ref_elv', 'u', 'v', 'w', 'q_CH4']
    """
    n_timesteps = x.shape[2]
    n_samples = x.shape[0]

    pathmax_value = pathmax(x_width, y_width, factor_x, factor_y)

    complete_array = []

    for i in range(n_samples): # this loop might be unessary
        # print('sample numnber', i)

        # append to some lists
        backtrack_u_ = []
        backtrack_v_ = []
        max_idx_ = []
        ch4_matrix = []

        ## U & V time series extracted from met sensor
        u = x[i][[0], :, 4]
        v = x[i][[0], :, 5]
        
        # pulling met sensor location
        # right now this is not an issue because the met sensor is 0,0, will need to think about when we add a second one
        met_dist = x[i][[0], :, 0].T[0]
        met_azi_sin  = x[i][[0], :, 1].T[0]
        met_azi_cos  = x[i][[0], :, 2].T[0]
        x_met, y_met = polar_to_cartesian(met_dist, met_azi_sin, met_azi_cos)

        # pulling the information from the ch4 sensors, distance, azi_sin and azi_cos and ch4
        ref_dist = x[i][[1, 2, 3], :, 0].T[0]
        azi_sin  = x[i][[1, 2, 3], :, 1].T[0]
        azi_cos  = x[i][[1, 2, 3], :, 2].T[0]
        ch4_data_ts = x[i][[1, 2, 3], :, 7]

        x_, y_ = polar_to_cartesian(ref_dist, azi_sin, azi_cos)
        ref_elevation  = x[i][[1, 2, 3], :, 3].T[0]
        stacked_data = np.column_stack((x_, y_, ref_elevation))
        # This variable is 3 by 9 
        repeated_pos = np.repeat(stacked_data, num_sensors).reshape(9,num_sensors).T

        ## emissions data
        for q in range(num_sensors):
            # print('sensor number', q+1)
            
            ch4_data = x[i][[1, 2, 3], :, 7][q]
            #print(ch4_data.shape)
            # findmaxch4
            max_c, time_max_c, max_idx = findmaxCH4(ch4_data, np.arange(n_timesteps)) 
            #print('max_c, max_idx, time_max_c')
            #print(max_c, max_idx, time_max_c)
            backtrack_u, backtrack_v = backtrack(ijk_start=time_max_c, u_sonic=u.ravel(), v_sonic=v.ravel(),
                                                 dt=dt, sensor_x=x_met[0], sensor_y=y_met[0], pathmax=pathmax_value)

            # append
            backtrack_u_.append(backtrack_u)
            backtrack_v_.append(backtrack_v)
            max_idx_.append(max_idx)
        
        ch4_matrix.append(ch4_data_ts[:, max_idx_].T)
        ch4_matrix_array = np.array(ch4_matrix).squeeze()
        backtrack_u_array = np.array(backtrack_u_).reshape(3, 1)
        backtrack_v_array = np.array(backtrack_v_).reshape(3, 1)
        merged_array = np.concatenate((backtrack_u_array, backtrack_v_array, repeated_pos, ch4_matrix_array), axis=1)
        complete_array.append(merged_array)
    
    # Goal to export:
    # backtrack_u, backtrack_v, x, y, z, x1, y1, z1, x2, y2, z2, ch4, ch4-1, ch4-2
    # that three times (or number of times of sensors)
        
    # need to fix some of these hard-codeded variables
    export_array =  np.array(complete_array).reshape(n_samples * num_sensors, 14)   
    
    print('shape of export array:', export_array.shape)
    return export_array

def mlp_target_output(y, target, number_of_sensors=3):
    # creates the x, y, z export of the 'true leak'
    # This does not include leak rate
    # y = decoder input 
    export_array = []

    num_sensors_int = np.int64(number_of_sensors)
    winners = np.argmax(target.squeeze(), axis=1)
    num_samples = np.int64(winners.shape)
    
    print('number of samples', num_samples)
    # figuring out which one is the leaky one
    for q in np.arange(len(winners)):
        leak_ = winners[q]
        # print('leak:', leak_)
        # pulling that data out
        # print("Y SHAPE", y.shape)
        # print("TARGET SHAPE:", target.shape)
        dist_sin_cos_elevation = y[q][leak_][:4].ravel()
    
        x_, y_ = polar_to_cartesian(dist_sin_cos_elevation[0], 
                                    dist_sin_cos_elevation[1], 
                                    dist_sin_cos_elevation[2])

        z_ = dist_sin_cos_elevation[3]
        row_array = np.asarray([x_, y_, z_])
        repeated_array = np.tile(row_array, (number_of_sensors, 1))
        export_array.append(repeated_array)
    
    reshape_size = num_sensors_int*num_samples
    export_array = np.array(export_array).reshape(np.int64(reshape_size[0]), num_sensors_int)
    print('EXPORT ARRAY:', export_array.shape)
    return export_array

def argmin_mlp_eval(y, mlp_output):
    # y = decoder input 
    # mlp_output = predicted xyz
    export_array = []
    y_squeezed = y.squeeze()
    
    if mlp_output.shape[0] != y.squeeze().shape[0]:
        raise ValueError("Arrays must have the same length (number of samples)")

    print('Number of samples:', y_squeezed.shape[0])
    print('Max number of leaks:', y_squeezed.shape[1])

    for q in np.arange(y_squeezed.shape[0]):
        
        ref_dist = y[q][:, :, 0].T[0]
        azi_sin  = y[q][:, :, 1].T[0]
        azi_cos  = y[q][:, :, 2].T[0]
        ref_elevation  = y[q][:, :, 3].T[0]
        ## 
        result = np.column_stack((ref_dist, azi_sin, azi_cos, ref_elevation))
        dropped_array = remove_all_rows_with_val(result, value_to_drop=-1)

        # 
        x_, y_ = polar_to_cartesian(dropped_array[:, 0], dropped_array[:, 1], dropped_array[:, 2])
        z_ = dropped_array[:, 3]
        xyz_ = np.column_stack((x_, y_ ,z_))
        
        geo = GeoCalculator(np.expand_dims(mlp_output[1], axis=1).T, xyz_ )
        distance = geo.distance_between_points_3d()
        
        arg_min = np.argmin(distance)
        
        zeros_array = np.zeros((y_squeezed.shape[1], 1))
        zeros_array[arg_min] = 1

        export_array.append(zeros_array.T)  
    print('Export shape',np.asarray(export_array).shape)      
    return np.asarray(export_array)