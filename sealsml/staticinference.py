# Data manipulation and analysis
import numpy as np
from numpy.typing import NDArray
import xarray as xr
from typing import List

# seals geo stuff
from sealsml.geometry import get_relative_azimuth

def load_inference(dataset, sitemap, timestep: int) -> NDArray:
    """
    Loads an netCDF from 'real' data, processes it into wind relative coordinates,
    and chunks it into the correct timestep length for the ML model. Also loads the sitemap to use as potential leaks.

    Parameters:
        dataset: The dataset to be processed.
        sitemap: Sitemap in netCDF
        timestep (int): The length of each timestep to chunk the data for ML inference.

    Returns:
        Encoder and Decoder for testing only (not training)
    """
    # loading the netCDF files here, this can be modified in the future if you want to load numpy arrays directly
    ds = xr.open_dataset(dataset)
    sitemap = xr.open_dataset(sitemap)

    XYZ_met = ds['metPos'].values

    u_met = ds.metVels.sel(metSensors=0).values.T[:, 0]
    v_met = ds.metVels.sel(metSensors=0).values.T[:, 1]
    w_met = ds.metVels.sel(metSensors=0).values.T[:, 2]

    XYZ_ch4 = ds['CH4Pos'].values

    encoder_arrays: List[NDArray] = []  # List to store encoder arrays for each iteration
    target_list: List[NDArray] = []  # List to store target arrays for each iteration

    for i in ds.CH4Sensors.values:
      # get_relative_azimuth(u, v, x_ref, y_ref, z_ref, x_target, y_target, z_target, time_series=True):
      output = get_relative_azimuth(
                        u_met, # u
                        v_met, # v
                        XYZ_met[0][0], #x_ref 
                        XYZ_met[0][1], #y_ref
                        XYZ_met[0][2], #z_ref
                        XYZ_ch4[i][0], #x_target
                        XYZ_ch4[i][1], #y_target
                        XYZ_ch4[i][2], #z_target
                        time_series=True  
                           )

      ch4_data = ds['q_CH4'].values[i]
      encoder_array = np.vstack((output, w_met, ch4_data))
      encoder_array = np.expand_dims(encoder_array, axis=-1)
      encoder_arrays.append(encoder_array)
      returned_array = np.concatenate(encoder_arrays, axis=-1).transpose(0, 2, 1)

    # Determine the number of complete timeseries that can be extracted
    total_length = returned_array.shape[2]
    num_complete_series = total_length // timestep

    # Trim the excess elements to ensure array dimensions align with complete time steps
    trimmed_length = timestep * num_complete_series
    trimmed_array = returned_array[:, :, :trimmed_length]

    # Print a message if the trimmed length is less than the original length
    if trimmed_length < total_length:
      print(f"Trimmed the array from original length {total_length} to {trimmed_length}")
      print(f"Number of elements dropped: {total_length - trimmed_length}")

    # Reshape the array to (variables, number of ch4 sensors, timeseries, number of timeseries)
    reshaped_array = trimmed_array.reshape(8, 4, timestep, num_complete_series).transpose(3, 1, 0, 2)
    
    #### Let's make some targets
    mask = sitemap['structureMask'].where(sitemap['structureMask'] == 1, drop=False).notnull()
    
    leak_x = sitemap.xPos.values.ravel()[mask.values.ravel()]
    leak_y = sitemap.yPos.values.ravel()[mask.values.ravel()]
    leak_z = sitemap.zPos.values.ravel()[mask.values.ravel()]
    
    print('Number of possible leaks:', len(leak_z))
    for b in range(num_complete_series):
      for a in range(len(leak_z )):
        targets = get_relative_azimuth(
                        u_met[b*timestep:(b+1)*timestep], # u
                        v_met[b*timestep:(b+1)*timestep], # v
                        XYZ_met[0][0], #x_ref 
                        XYZ_met[0][1], #y_ref
                        XYZ_met[0][2], #z_ref
                        leak_x[a], #x_target
                        leak_y[a], #y_target
                        leak_z[a], #z_target
                        time_series=False  
                           )
        target_list.append(targets)
    
    # Target array is currently number of targets, variables, time)
    print('Number of samples:', num_complete_series )    
    target_arrays = np.array(target_list, dtype=float).reshape(num_complete_series, len(leak_z), 6, 1)
    # returns encoder and decoder as multi-dim numpy arrays
    return reshaped_array, target_arrays