import numpy as np
from bridgescaler import DQuantileScaler,save_scaler
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

    # Check if factors are within the valid range [0, 0.90]
    if not (0 <= np.min(factor_x) <= 5.0) or not (0 <= np.min(factor_y) <= 5.0):
        raise ValueError("Factors should range between 0 and 5.0.")

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

def backtrack(ijk_start: int, u_sonic, v_sonic, dt, sensor_x, sensor_y, pathmax, n_pot_leaks, leak_x, leak_y, structure_flag):
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
    distances=np.zeros(shape=(n_pot_leaks))
    closest_i=np.zeros(shape=(ijk_start))
    leak_id_i=np.zeros(shape=(ijk_start))
    closest_u_avg=np.zeros(shape=(ijk_start))
    closest_v_avg=np.zeros(shape=(ijk_start))
    xn_i=np.zeros(shape=(ijk_start+1))
    yn_i=np.zeros(shape=(ijk_start+1))
    counter=-1
    xn_i[0]=xn
    yn_i[0]=yn

    # Backtrack along the velocity path; dt is constant

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

        step_dist=np.sqrt((xn-xnm1)**2+(yn-ynm1)**2)

        xn = xnm1
        yn = ynm1

        # Calculating straight-line Distance
        dx = sensor_x - xn
        dy = sensor_y - yn
        # straight line distance, doesn't account for meandering
        total_st_dist = np.sqrt((dx**2 + dy**2))
        # path distancee
        total_dist +=step_dist

        denominator = max(1, (ijk_start - ijk))

        if structure_flag == 1:

            counter += 1
            distances=np.sqrt((xn-leak_x)**2+(yn-leak_y)**2)
            closest_i[counter]=np.min(distances)
            leak_id_i[counter]=np.argmin(distances)
            closest_u_avg[counter]=ux_sum/denominator
            closest_v_avg[counter]=vy_sum/denominator

    if structure_flag == 1:

        closest=closest_i[0]
        closest_leak_id=0
        leak_counter=0
        for j in range(1,counter+1):
            if closest_i[j] < closest:
                closest=closest_i[j]
                closest_leak_id=int(leak_id_i[j])
                leak_counter=j

        avg_u=closest_u_avg[leak_counter]
        avg_v=closest_v_avg[leak_counter]

    else:

        # Compute average horizontal wind components

        avg_u = ux_sum / denominator
        avg_v = vy_sum / denominator

    return avg_u, avg_v, xn, yn

def backtrack_preprocess(data, n_sensors=3, x_width=40, y_width=40, factor_x=0.4, factor_y=0.4, structure_flag=1, dtc=1.0, verbose_log=False):

    # This function creates both the input data, and target data for the ANN/MLP

    encoder = data['encoder_input'].load()
    decoder = data['decoder_input'].load()
    n_samples = encoder.shape[0]
    n_timesteps = encoder.shape[2]

    #     decoder_input  (sample, pot_leak, target_time, variable, mask)
    n_pot_leaks=len(decoder[0,:,0,0,0])-1
    print('n_pot_leaks = ',n_pot_leaks)
    x_pot_leaks=np.zeros(shape=(n_samples,n_pot_leaks))
    y_pot_leaks=np.zeros(shape=(n_samples,n_pot_leaks))
    z_pot_leaks=np.zeros(shape=(n_samples,n_pot_leaks))
    x_pot_leaks, y_pot_leaks = polar_to_cartesian(decoder[:,:,0,0,0],
                                                  decoder[:,:,0,1,0],
                                                  decoder[:,:,0,2,0])
    z_pot_leaks = decoder[:,:,0,0,0]*np.sin(decoder[:,:,0,3,0])

    print('pot_leaks wind-relative cartesian coords set!')
    # This statement collapses all CH4 sensor information into one input line rather than n lines, 
    # where n_sensors = number of CH4 sensors
    # The input layer (per sample) has length (n_sensors)*(5+(nsensors)) 
    # so n_sensors * (u,v,x,y,z and n_sensors of CH4 values)
    # The n_sensors replicates correspond to the "max" window around each sensor's max CH4 observation
    
    input_array = np.zeros(shape=(n_samples, (n_sensors+5) * n_sensors))
    
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
    print('Key variables extracted!')

    if verbose_log:
        print('x_sensor= \n',x_sensor[10:21,:],'\n y_sensor= \n',y_sensor[10:21,:])

    met_times = np.linspace(0.0,np.float32(n_timesteps-1)*dtc,n_timesteps,endpoint=True)

    if x_width < y_width:
        L_scale=y_width
    else:
        L_scale=x_width

    H_scale=15.

    speed=np.zeros(shape=(n_samples))

    # This slices from 1 to n_sensors + 1, the met is the 0th sensor 
    ch4_time_series = encoder.sel(sensor=slice(1, n_sensors + 1),
                                  variable='q_CH4',
                                  mask=0)

    print('ch4.time.series.shape=',ch4_time_series.shape)
    print('u.shape=',u.shape,'v.shape=',v.shape)

    for i in range(n_samples):
        ch4 = []
        coords = []
        u_backtrack = []
        v_backtrack = []

        ui = u.isel(sample=i).values.ravel()
        vi = v.isel(sample=i).values.ravel()

        ui_avg=np.mean(ui)
        vi_avg=np.mean(vi)

        speed[i]=np.max(np.sqrt(ui**2+vi**2))

        for s in range(0,n_sensors):
            sensor_time_series = ch4_time_series[i, s].values
            max_CH4, time, idx = findmaxCH4(sensor_time_series, met_times) 
            if max_CH4 > 1.0e-9:
                backtrack_u, backtrack_v, x_last_back, y_last_back = backtrack(ijk_start=idx,
                                                 u_sonic=ui,
                                                 v_sonic=vi,
                                                 dt=dtc,
                                                 sensor_x=x_sensor[i,s],
                                                 sensor_y=y_sensor[i,s],
                                                 pathmax=pathmax_value,
                                                 n_pot_leaks=n_pot_leaks,
                                                 leak_x=x_pot_leaks[i,:],
                                                 leak_y=y_pot_leaks[i,:],
                                                 structure_flag=structure_flag)
            else:
                backtrack_u=ui_avg
                backtrack_v=vi_avg

            u_backtrack.append(backtrack_u)
            v_backtrack.append(backtrack_v)
            coords.append(x_sensor[i,s])
            coords.append(y_sensor[i,s])
            # vertical displacement relative to met sensor height
            coords.append(relative_sensor_locs.sel(variable='ref_distance').values[i,s]*
                          np.sin(relative_sensor_locs.sel(variable='ref_elv').values[i, s]))

            # this appends the ch4 values at all the other sensors r at the time of the max CH4 value at sensor s
            for r in range(0,n_sensors):  

                if r != s:
                    ch4.append(ch4_time_series[i,r].values[idx])
                else:
                    ch4.append(max_CH4)

        input_array[i] = np.array(u_backtrack + v_backtrack + coords + ch4)

    print(input_array.shape)
    if verbose_log:
        print('input_array=\n',input_array[10:21,:])
    
    return input_array, speed, L_scale, H_scale, n_samples, n_pot_leaks, x_pot_leaks, y_pot_leaks, z_pot_leaks

def truth_values(data):

    encoder = data['encoder_input'].load()
    decoder = data['decoder_input'].load()
    n_samples = encoder.shape[0]
    target = data['target'].values
    target_polar=np.zeros(shape=(n_samples,4))
    target_array=np.zeros(shape=(n_samples,4))

    indices_tuple=np.nonzero(target[:,:,0])

#   assumes met sensor is origin in all 3 directions
    for s in range(indices_tuple[0].shape[0]):
        target_polar[s,:4]=decoder[s,indices_tuple[1][s],0,:4,0]

    x_leak, y_leak = polar_to_cartesian(target_polar[:,0],
                                        target_polar[:,1],
                                        target_polar[:,2])
    z_leak=target_polar[:,0]*np.sin(target_polar[:,3])

#   target_array = np.concatenate([leak_locs,met_locs, data['leak_rate'].values.reshape(-1, 1)], axis=1)
    target_array[:,0]= x_leak
    target_array[:,1]= y_leak
    target_array[:,2]= z_leak
    target_array[:,3]=data['leak_rate'].values

    return target_array

def create_binary_preds_relative(data, y_pred: np.ndarray, ranked=False) -> np.ndarray:
    '''
    Create either a binary (0,1) or ranked array based on predicted coordinates and potential leak locations.

    Returns:
    - location_array: np.ndarray, shape (n_samples, max_potential_leaks), binary or ranked array.
        If ranked=True, ranks distances from 1 (closest) to n (farthest) for each potential leak location.
            The value of zero marks nothing basically, for 10 leak locations it will range from 1 to 11, and then it will use 0 to mark nothing
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

def backtrack_scaleDataTuple(data_tuple, scaler=None, fit_scaler=False):
    ret_tuple=()
    if fit_scaler == True:
       scaler = DQuantileScaler()
       scaled_encoder = scaler.fit(data_tuple[0])
    for i in range(len(data_tuple)):
       scaled_data = scaler.transform(data_tuple[i])
       ret_tuple = ret_tuple + (scaled_data,) 
    return ret_tuple, scaler

def backtrack_unscaleDataTuple(data_tuple, scaler):
    ret_tuple=()
    for i in range(len(data_tuple)):
       unscaled_data = scaler.inverse_transform(data_tuple[i])
       ret_tuple = ret_tuple + (unscaled_data,) 
    return ret_tuple

def mapPredlocsToClosestPL(pred_array, pl_x, pl_y, pl_z, use3dDist=False):
    ret_array = np.array(pred_array)
    for s in range(pred_array.shape[0]):
        if use3dDist:
            distances=np.sqrt( (pred_array[s,0]-pl_x[s,:])**2 + (pred_array[s,1]-pl_y[s,:])**2 + (pred_array[s,2]-pl_z[s,:])**2)
        else:
          distances=np.sqrt( (pred_array[s,0]-pl_x[s,:])**2 + (pred_array[s,1]-pl_y[s,:])**2 )
        ret_array[s,0]=pl_x[s,np.argmin(distances)]
        ret_array[s,1]=pl_y[s,np.argmin(distances)]
        ret_array[s,2]=pl_z[s,np.argmin(distances)]
    return ret_array

def scalings_bjt(x,y,speed,L_scale,H_scale,n_records,n_width,n_sensors):

    # scaling factors, in order:
    # 0. x_ref - this is 0 since variables are already relative to reference point (the met location)
    # 1. y_ref - this is 0 since variables are already relative to reference point (the met location)
    # 2. z_ref - this is 0 since z starts at 0.
    # 3. L_scale - maximum horizontal scale
    # 4. H_scale - maximum vertical scale
    # 5. max_speed - maximum speed used for scaling velocities
    # 6. CH4 background - background CH4 level 
    # 7. CH4 maximum - maximum CH4 value from all CH4 sensors
    # 8. CH4_scale = log(max(CH4maximum,CH4.background)/CH4.background)
    # 9. Q_min = minimum leak rate - to prevent log10(0.) if no leak
    # 10. Q_max = maximum leak rate in training data
    # 11. Q_scale = scaling for Q (leak rate) = log10((Qmax+Qmin)/Qmin)

    scaling_factors=np.zeros(shape=(12))

    scaling_factors[0]=0.
    scaling_factors[1]=0.
    scaling_factors[2]=0.
    scaling_factors[3]=L_scale
    scaling_factors[4]=H_scale
    scaling_factors[5]=np.max(speed)
    scaling_factors[6]=1.e-12  # 1.e-6, but LES sims didn't have background, and here, just used to prevent log(0.)
    n5=n_sensors*5
    scaling_factors[7]=0.
    for i in range(0,n_records):
        for s in range(0,n_sensors):
            sc7=x[i][n5+s+s*n_sensors]
            if sc7 > scaling_factors[7]:
                scaling_factors[7]=sc7
    scaling_factors[8]=np.log10(max(scaling_factors[7],scaling_factors[6])/scaling_factors[6])
    scaling_factors[9]=1.0e-10
    scaling_factors[10]=np.max(y[:][3])
    scaling_factors[11]=np.log10((scaling_factors[10]+scaling_factors[9])/scaling_factors[9])

    print('scaling_factors= \n',scaling_factors)

    return scaling_factors

def scaler_bjt_x(x,scaling_factors):

    n_records=x.shape[0]
    n_width=x.shape[1]
    n_sensors=int((-5.+np.sqrt(25.+4.*float(n_width)))/2.)
    print('in scaler_bjt_x, n_records,n_sensors=',n_records,n_sensors)
   
    x_ref=scaling_factors[0]
    y_ref=scaling_factors[1]
    z_ref=scaling_factors[2]
    L_scale=scaling_factors[3]
    H_scale=scaling_factors[4]
    max_speed=scaling_factors[5]
    CH4_bckgrd=scaling_factors[6]
    CH4_scale=scaling_factors[8]

    x_scaled=np.zeros(shape=(n_records,n_width))
    n5s=5*n_sensors
    for s in range(0,n_records):
        for i in range(0,n_sensors):
            x_scaled[s][i]=x[s][i]/max_speed
            x_scaled[s][i+n_sensors]=x[s][i+n_sensors]/max_speed
            x_scaled[s][2*n_sensors+3*i]=0.5*(1.+(x[s][2*n_sensors+3*i]-x_ref)/L_scale)
            x_scaled[s][2*n_sensors+3*i+1]=0.5*(1.+(x[s][2*n_sensors+3*i+1]-y_ref)/L_scale)
            x_scaled[s][2*n_sensors+3*i+2]=(x[s][2*n_sensors+3*i+2]-z_ref)/H_scale
            nsi=n_sensors*i
            for j in range(0,n_sensors):
                x_scaled[s][n5s+nsi+j]=np.log10(max(x[s][n5s+nsi+j],CH4_bckgrd)/CH4_bckgrd)/CH4_scale
    
    return x_scaled

def scaler_bjt_y(y,scaling_factors):
    
    n_records=y.shape[0]
    n_width=4
    x_ref=scaling_factors[0]
    y_ref=scaling_factors[1]
    z_ref=scaling_factors[2]
    L_scale=scaling_factors[3]
    H_scale=scaling_factors[4]
    Q_min=scaling_factors[9]
    Q_scale=scaling_factors[11]

    y_scaled=np.zeros(shape=(n_records,n_width))
    for s in range(0,n_records):
        y_scaled[s][0]=0.5*(1.+(y[s][0]-x_ref)/L_scale)
        y_scaled[s][1]=0.5*(1.+(y[s][1]-y_ref)/L_scale)
        y_scaled[s][2]=(y[s][2]-z_ref)/H_scale
        y_scaled[s][3]=np.log10((y[s][3]+Q_min)/Q_min)/Q_scale
   
    return y_scaled

def scaler_bjt_y_inverse(y,scaling_factors):
    
    n_records=y.shape[0]
    n_width=4
    x_ref=scaling_factors[0]
    y_ref=scaling_factors[1]
    z_ref=scaling_factors[2]
    L_scale=scaling_factors[3]
    H_scale=scaling_factors[4]
    Q_min=scaling_factors[9]
    Q_scale=scaling_factors[11]

    y_unscaled=np.zeros(shape=(n_records,n_width))
    for s in range(0,n_records):
        y_unscaled[s][0]=(2.*y[s][0]-1.)*L_scale+x_ref
        y_unscaled[s][1]=(2.*y[s][1]-1.)*L_scale+y_ref
        y_unscaled[s][2]=y[s][2]*H_scale+z_ref
        y_unscaled[s][3]=10.**(y[s][3]*Q_scale)*Q_min-Q_min

    return y_unscaled

