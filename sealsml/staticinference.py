# Data manipulation and analysis
import numpy as np
from numpy.typing import NDArray
import xarray as xr

# seals geo stuff
from sealsml.geometry import get_relative_azimuth

def load_inference(dataset: xr.Dataset, timestep: int) -> NDArray:
    
    ds = xr.open_dataset(dataset)
    ##
    XYZ_met = ds['metPos'].values

    u_met = ds.metVels.sel(metSensors=0).values.T[:, 0]
    v_met = ds.metVels.sel(metSensors=0).values.T[:, 1]
    w_met = ds.metVels.sel(metSensors=0).values.T[:, 2]

    XYZ_ch4 = ds['CH4Pos'].values

    encoder_arrays = []  # List to store encoder arrays for each iteration

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
      returned_array = np.concatenate(encoder_arrays, axis=-1)
    return returned_array