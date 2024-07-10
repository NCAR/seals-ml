import numpy as np
from typing import Tuple
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
    # """
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

def backtrack_preprocess(data, n_sensors=3, x_width=40, y_width=40, factor_x=0.4, factor_y=0.4):
    # This function creates both the input data, and target data for the ANN/MLP
    encoder = data['encoder_input'].load()
    targets = data['target'].values
    target_mask = np.argwhere(targets == 1)
    leak_locs = data['leak_meta'].values[target_mask[:, 0], target_mask[:, 1]]
    met_locs = data['met_sensor_loc'].values
    n_samples = encoder.shape[0]
    n_timesteps = encoder.shape[2]

    print('leak_locs=',leak_locs)
    print('met_locs=',met_locs)

    # This statement collapses all CH4 sensor information into one input line rather than n lines, 
    # where n_sensors = number of CH4 sensors
    # The input layer (per sample) has length (n_sensors)*(5+(nsensors)) 
    # so n_sensors * (u,v,x,y,z and n_sensors of CH4 values)
    # The n_sensors replicates correspond to the "max" window around each sensor's max CH4 observation
    input_array = np.zeros(shape=(n_samples, n_sensors * (5+n_sensors)))  
    
    target_array = np.concatenate([met_locs - leak_locs, data['leak_rate'].values.reshape(-1, 1)], axis=1)
    pathmax_value = pathmax(x_width=x_width, y_width=y_width, factor_x=factor_x, factor_y=factor_y)
    u = encoder.sel(sensor=0,
                    variable=('u'),
                    mask=0)
    v = encoder.sel(sensor=0,
                    variable=('v'),
                    mask=0)
    # This slices it from 1 to n_sensors + 1, the met is the 0th sensor 
    relative_sensor_locs = encoder.sel(sensor=slice(1, n_sensors + 1),
                                       time=0,
                                       variable=['ref_distance', 'ref_azi_sin', 'ref_azi_cos', 'ref_elv'],
                                       mask=0)
    x_sensor, y_sensor = polar_to_cartesian(relative_sensor_locs.sel(variable='ref_distance'),
                              relative_sensor_locs.sel(variable='ref_azi_sin'),
                              relative_sensor_locs.sel(variable='ref_azi_cos'))

    met_locs = encoder.sel(sensor=0,
                           time=0,
                           variable=['ref_distance', 'ref_azi_sin', 'ref_azi_cos'],
                           mask=0)
    x_met, y_met = polar_to_cartesian(met_locs.sel(variable='ref_distance'),
                                      met_locs.sel(variable='ref_azi_sin'),
                                      met_locs.sel(variable='ref_azi_cos'))
    
    # This slices from 1 to n_sensors + 1, the met is the 0th sensor 
    ch4_time_series = encoder.sel(sensor=slice(1, n_sensors + 1),
                                  variable='q_CH4',
                                  mask=0)
    for i in range(n_samples):
        ch4 = []
        coords = []
        u_backtrack = []
        v_backtrack = []

        ui = u.isel(sample=i).values.ravel()
        vi = v.isel(sample=i).values.ravel()

        for s in range(0,n_sensors):
            sensor_time_series = ch4_time_series[i, s].values
            max_CH4, time, idx = findmaxCH4(sensor_time_series, np.arange(n_timesteps))
            backtrack_u, backtrack_v = backtrack(ijk_start=idx,
                                                 u_sonic=ui,
                                                 v_sonic=vi,
                                                 dt=1,
                                                 sensor_x=x_sensor[i,s],
                                                 sensor_y=y_sensor[i,s],
                                                 pathmax=pathmax_value)
            u_backtrack.append(backtrack_u)
            v_backtrack.append(backtrack_v)
            coords.append(x_sensor[i,s])
            coords.append(y_sensor[i,s])
            coords.append(relative_sensor_locs.sel(variable='ref_elv').values[i, s])

#           this appends the ch4 values at all the other sensors r at the time of the max CH4 value at sensor s

            for r in range(0,n_sensors):  

                if r != s:
                    ch4.append(ch4_time_series[i,r].values[idx])
                else:
                    ch4.append(max_CH4)

        input_array[i] = np.array(u_backtrack + v_backtrack + coords + ch4)

        print('i,u_backtrack,v_backtrack,coords,ch4=','\n',i,'\n',u_backtrack,'\n',v_backtrack,'\n',coords,'\n',ch4)


    return input_array, target_array

def create_binary_preds_relative(data, y_pred: np.ndarray, ranked=False) -> np.ndarray:
    '''
    Create either a binary (0,1) or ranked array based on predicted coordinates and potential leak locations.

    Returns:
    - location_array: np.ndarray, shape (n_samples, max_potential_leaks), binary or ranked array.
      If ranked=True, ranks distances from 1 (closest) to n (farthest) for each potential leak location.
      If ranked=False, marks the closest leak location with 1 (binary classification).
    '''
    n_samples = y_pred.shape[0]
    y_true = data['leak_meta'].values
    met_locs = data['met_sensor_loc'].values
    xyz_pred = y_pred[:, :3]
    location_array = np.zeros(shape=y_true.shape[:-1])
    
    # Loop through each sample
    for s in range(n_samples):
        # Remove rows where leak_meta is zero (indicating no leak)
        pot_leak_locs = remove_all_rows_with_val(y_true[s], value_to_drop=0)
        pred_coord = xyz_pred[s]
        geo = GeoCalculator(pred_coord, met_locs[s] - pot_leak_locs)
        distance = geo.distance_between_points_3d()
        
        if ranked:
            # Rank distances and assign ranks to location_array
            ranked_indices = np.argsort(distance) # Computes indices that would sort the distance array in ascending order.
            # Index of the location in location_array corresponding to the sorted order of distances.
            for rank, idx in enumerate(ranked_indices, start=1):
                location_array[s, idx] = rank
        else:
            # Mark the closest leak location with 1 in location_array
            arg_min = np.argmin(distance)
            location_array[s, arg_min] = 1

    return location_array
