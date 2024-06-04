# Data manipulation and analysis
import numpy as np
import xarray as xr
from typing import List
from numpy.typing import NDArray
import yaml
import os

# seals geo stuff
from sealsml.geometry import get_relative_azimuth

def extract_ts_segments(time_series, time_window_size:int, window_stride:int):
    """
    Extract segments from a time series array.

    Parameters:
    - time_series (numpy.ndarray): The input time series.
    - segment_length (int): The length of each segment (must be an integer).
    - stride (int): The stride between consecutive segments (must be an integer).

    Returns:
    - start_end_indices (numpy.ndarray): An array containing the start and end indices of each segment.
    - dropped_elements (numpy.ndarray): Any elements that are dropped because they don't fit into a full segment.
    """
    num_segments = (len(time_series) - time_window_size) // window_stride + 1
    print('Number of time series segments:', num_segments)
    start_end_indices = np.zeros((num_segments, 2), dtype=int)

    for i in range(num_segments):
        start_index = i * window_stride
        end_index = start_index + time_window_size
        start_end_indices[i] = [start_index, end_index]

    last_end_index = start_end_indices[-1, 1]
    dropped_elements = time_series[last_end_index:]
    print('Number of dropped elements:', np.size(dropped_elements))
    return start_end_indices, dropped_elements

def specific_site_data_generation(dataset_path, sitemap_path, time_window_size: int, window_stride:int, export_mean_wd = False):
  """
  This is not for use with fully 3D LES cubes of data. This assumes n number of sensors and some site information. 

  Typical use case might be for interence on real data.
  
  Loads an netCDF from 'real' data, processes it into wind relative coordinates,
  and chunks it into the correct timestep length for the ML model. Also loads the sitemap to use as potential leaks.

  Parameters as netCDF:
    dataset: The dataset to be processed.
    sitemap: Sitemap in netCDF
    timestep (int): The length of each timestep to chunk the data for ML inference.

  Returns:
    Encoder and Decoder in a xarray dataset
  """
   
  ds = xr.open_dataset(dataset_path).load()
  sitemap = xr.open_dataset(sitemap_path).load()

  encoder_arrays: List[NDArray] = []  # List to store encoder arrays for each iteration
  target_list: List[NDArray] = []  # List to store target arrays for each iteration

  # xyz location of the sensors does not change with time
  XYZ_met = ds['metPos'].values
  XYZ_ch4 = ds['CH4Pos'].values
  print('How many CH4 sensors?', len(ds.CH4Sensors.values))

  #### Let's make some targets
  mask = sitemap['structureMask'].where(sitemap['structureMask'] == 1, drop=False).notnull()
    
  leak_x = sitemap.xPos.values.ravel()[mask.values.ravel()]
  leak_y = sitemap.yPos.values.ravel()[mask.values.ravel()]
  leak_z = sitemap.zPos.values.ravel()[mask.values.ravel()]
  print('Number of possible leaks:', len(leak_z))

  # time series chunking
  ts_indicies, dropped = extract_ts_segments(ds.time.values, 
                                             time_window_size=time_window_size, 
                                             window_stride=window_stride)

  # we are going to run a loop for each 
  for t in range(ts_indicies.shape[0]):
    start = ts_indicies[t][0]
    end = ts_indicies[t][1]
  
    # new dataset 
    ds_chunked = ds.isel(time=slice(start, end))

    u_met = ds_chunked.metVels.sel(metSensors=0).values.T[:, 0]
    v_met = ds_chunked.metVels.sel(metSensors=0).values.T[:, 1]
    w_met = ds_chunked.metVels.sel(metSensors=0).values.T[:, 2]
  
    # For decoder
    for a in range(len(leak_z)):
      targets, mean_wd = get_relative_azimuth(
        u_met, # u
        v_met, # v
        XYZ_met[0][0], #x_ref 
        XYZ_met[0][1], #y_ref
        XYZ_met[0][2], #z_ref
        leak_x[a], #x_target
        leak_y[a], #y_target
        leak_z[a], #z_target
        time_series=False
        )
      target_list.append(targets[:4]) # This is the other line that needs to be fixed

    # For encoder
    for i in ds.CH4Sensors.values:
    # get_relative_azimuth(u, v, x_ref, y_ref, z_ref, x_target, y_target, z_target, time_series=True):
      output, mean_wd = get_relative_azimuth(
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

      ch4_data = ds_chunked['q_CH4'].values[i]
      encoder_array = np.vstack((output, w_met, ch4_data))
      encoder_array = np.expand_dims(encoder_array, axis=-1)
      encoder_arrays.append(encoder_array)
      returned_array = np.concatenate(encoder_arrays, axis=-1).transpose(0, 2, 1)

  # Reshape the array to (variables, number of ch4 sensors, timeseries, number of timeseries)
  encoder_output = returned_array.reshape(8, len(ds.CH4Sensors.values), time_window_size, ts_indicies.shape[0]).transpose(3, 1, 2, 0)

  # Target array is currently number of targets, variables, time)
  print('Target list shape' , np.array(target_list, dtype=float).shape)
  decoder_output = np.array(target_list, dtype=float).reshape(ts_indicies.shape[0], len(leak_z), 4, 1).transpose(0, 1, 3, 2)
  
  # Create xarray Dataset
  encoder_ds = xr.DataArray(encoder_output,
                            dims=['sample', 'sensor', 'time', 'variable'],
                            coords={'variable': ["ref_distance", "ref_azi_sin", "ref_azi_cos", "ref_elv", "u", "v", "w", "q_CH4"]},
                            name="encoder_input").astype('float32')

  decoder_ds = xr.DataArray(decoder_output,
                            dims=['sample', 'pot_leak', 'target_time', 'variable'],
                            coords={'variable': ["ref_distance", "ref_azi_sin", "ref_azi_cos", "ref_elv"]}, # I can add u,v,w, q_CH4 back
                            name="decoder_input").astype('float32')
    
  ds_static_output = xr.merge([encoder_ds, decoder_ds])
  # Decide what to export 
  if export_mean_wd:
    return ds_static_output, mean_wd
  else:
    return ds_static_output