# Data manipulation and analysis
import numpy as np
import xarray as xr
import numpy.matlib

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

def specific_site_data_generation(dataset_path, sitemap_path, potloc_path, time_window_size: int, window_stride:int,
                                  sensor_type_value=-999, emission_vars=("q_CH4"),
                                  met_vars=("u", "v", "w"),
                                  coord_vars=("ref_distance", "ref_azi_sin", "ref_azi_cos", "ref_elv")):
  """
  This is not for use with fully 3D LES cubes of data. This assumes n number of sensors and some site information. 

  Typical use case might be for interence on real data.
  
  Loads an netCDF from 'real' data, processes it into wind relative coordinates,
  and chunks it into the correct timestep length for the ML model. Also loads the sitemap to use as potential leaks.

  Parameters as netCDF:
    dataset: The dataset to be processed.
    sitemap: Sitemap in netCDF
    potloc_path: Potential leak locations file
  Configurable parameters:
    time_window_size (int): Length of timeseries (index based) for each sample [needs to be consitent with the ML model...]
    window_stride (int): stride width to slide window through time (index based)

  Returns:
    Encoder and Decoder in a xarray dataset
  """
  # variables
  variables = list(coord_vars) + list(met_vars) + list(emission_vars)

  ds = xr.open_dataset(dataset_path).load()
  sitemap = xr.open_dataset(sitemap_path).load()
  potloc = xr.open_dataset(potloc_path).load()

  # xyz location of the sensors 
  # does not change with time
  XYZ_met = ds['metPos'].values
  XYZ_ch4 = ds['CH4Pos'].values

  potleak_locs = potloc.srcPotLeakLocation.values
  leak_x = potleak_locs[:,0]
  leak_y = potleak_locs[:,1]
  leak_z = potleak_locs[:,2]

  # time series chunking
  ts_indicies, dropped = extract_ts_segments(ds.time.values, 
                                             time_window_size=time_window_size, 
                                             window_stride=window_stride)

  sample_len = ts_indicies.shape[0]
  sensor_len = ds.sizes['CH4Sensors'] + ds.sizes['metSensors']
  variable_len = len(variables)
  mask_len = 2
  pot_leak_len = potloc.sizes['plDim']
  target_time_len = 1

  encoder_input = np.zeros([sample_len,sensor_len,time_window_size,variable_len,mask_len])
  print('encoder_input.shape=',encoder_input.shape)
  decoder_input = np.zeros([sample_len,pot_leak_len,target_time_len,variable_len,mask_len])
  print('decoder_input.shape=',decoder_input.shape)
  mean_wd = np.zeros(sample_len)

  # encoder masks
  mask_met = np.isin(variables, emission_vars) * sensor_type_value
  mask_ch4 = np.isin(variables, met_vars) * sensor_type_value
  mask_met_2d = np.matlib.repmat(mask_met, time_window_size, 1)
  mask_ch4_2d = np.matlib.repmat(mask_ch4, time_window_size, 1)
  # decoder mask
  leak_mask = np.isin(variables, met_vars+emission_vars) * sensor_type_value

  metVels = ds.metVels.values
  q_CH4 = ds.q_CH4.values

  # we are going to run a loop for each
  for t in range(sample_len):
    start = ts_indicies[t][0]
    end = ts_indicies[t][1]
  
    # velocity components from the reference (and only) meteorological sensor [encoder]
    u_met = metVels[0,0,start:end]
    v_met = metVels[0,1,start:end]
    w_met = metVels[0,2,start:end]

    derived_vars, mean_wd_t = get_relative_azimuth(u=u_met,
                                                   v=v_met,
                                                   x_ref=XYZ_met[0][0],
                                                   y_ref=XYZ_met[0][1],
                                                   z_ref=XYZ_met[0][2],
                                                   x_target=XYZ_met[0][0],
                                                   y_target=XYZ_met[0][1],
                                                   z_target=XYZ_met[0][2],
                                                   time_series=True)

    encoder_input[t,0,:,0:6,0] = derived_vars.T
    encoder_input[t,0,:,6,0] = w_met
    encoder_input[t,0,:,:,1] = mask_met_2d
    mean_wd[t] = mean_wd_t

    # CH4 sensors [encoder]
    for i in ds.CH4Sensors.values:

        ch4_tmp = q_CH4[i,start:end]

        derived_vars, theta_wd = get_relative_azimuth(u=u_met,
                                                      v=v_met,
                                                      x_ref=XYZ_met[0][0],
                                                      y_ref=XYZ_met[0][1],
                                                      z_ref=XYZ_met[0][2],
                                                      x_target=XYZ_ch4[i][0],
                                                      y_target=XYZ_ch4[i][1],
                                                      z_target=XYZ_ch4[i][2],
                                                      time_series=True)

        encoder_input[t,i+ds.sizes['metSensors'],:,0:4,0] = derived_vars[0:4].T
        encoder_input[t,i+ds.sizes['metSensors'],:,7,0] = ch4_tmp
        encoder_input[t,i+ds.sizes['metSensors'],:,:,1] = mask_ch4_2d

    # Potential leaks [decoder]
    for a in range(len(leak_z)):

        derived_vars, theta_wd = get_relative_azimuth(u=u_met,
                                                      v=v_met,
                                                      x_ref=XYZ_met[0][0],
                                                      y_ref=XYZ_met[0][1],
                                                      z_ref=XYZ_met[0][2],
                                                      x_target=leak_x[a],
                                                      y_target=leak_y[a],
                                                      z_target=leak_z[a],
                                                      time_series=False)

        decoder_input[t,a,0,0:4,0] = np.squeeze(derived_vars[0:4])
        decoder_input[t,a,0,:,1] = leak_mask

  sensor_meta_1d = np.zeros([sensor_len,ds.sizes['locDim']])
  sensor_name = []
  for ss in range(0,sensor_len):
      if (ss==0):
          sensor_name.append(ds.metSensorsName.values[ss])
          for ll in range(0,3):
              sensor_meta_1d[ss,ll] = XYZ_met[0][ll]   
      else:
          sensor_name.append(ds.CH4SensorsName.values[ss-1])
          for ll in range(0,3):
              sensor_meta_1d[ss,ll] = XYZ_ch4[ss-1][ll]

  time_v = ds.time.values
  dt = (time_v[1]-time_v[0])/np.timedelta64(1000000000, 'ns')

  # Create xarray Dataset
  encoder_ds = xr.DataArray(encoder_input,
                                  dims=['sample', 'sensor', 'time', 'variable', 'mask'],
                                  coords={'variable': variables},
                                  name="encoder_input").astype('float32')

  decoder_ds = xr.DataArray(decoder_input,
                            dims=['sample', 'pot_leak', 'target_time', 'variable', 'mask'],
                            coords={'variable': variables},
                            name="decoder_input").astype('float32')

  mean_wd = xr.DataArray(mean_wd,
                            dims=['sample'],
                            coords={},
                            name="mean_wd").astype('float32')

  met_sensor_loc = xr.DataArray(np.tile(XYZ_met,[sample_len,1]),
                                dims=['sample', 'sensor_loc'],
                                coords={'sensor_loc': ['xPos', 'yPos', 'zPos']},
                                name="met_sensor_loc").astype('float32')

  sensor_meta = xr.DataArray(np.tile(sensor_meta_1d,[sample_len,1,1]),
                             dims=['sample', 'sensor', 'sensor_loc'],
                             coords={'sensor_loc': ['xPos', 'yPos', 'zPos']},
                             name="sensor_meta").astype('float32')

  sensor_name = xr.DataArray(sensor_name,
                             dims=['sensor'],
                             coords={},
                             name="sensor_name")

  dt = xr.DataArray(dt,
                    dims=[],
                    coords={},
                    name="dt").astype('float32')

  time_ref = xr.DataArray(time_v[0],
                          dims=[],
                          coords={},
                          name="time_ref")

  ds_static_output = xr.merge([encoder_ds, decoder_ds, mean_wd, met_sensor_loc, sensor_meta, sensor_name, dt, time_ref])
  return ds_static_output
