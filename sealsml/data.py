import xarray as xr
import pandas as pd
import numpy as np
from os.path import join, exists
from os import makedirs
from sealsml.geometry import GeoCalculator, get_relative_azimuth
from bridgescaler import DeepQuantileTransformer, DeepMinMaxScaler, DeepStandardScaler

class DataSampler(object):
    """ Sample LES data with various geometric configurations. """

    def __init__(self, min_trace_sensors=3, max_trace_sensors=15, min_leak_loc=1, max_leak_loc=10,
                 sensor_height_min=1,
                 sensor_height_max=4, 
                 leak_height_min=0, 
                 leak_height_max=4, 
                 sensor_type_mask=1, sensor_exist_mask=-1,
                 coord_vars=["ref_distance", "ref_azi_sin", "ref_azi_cos", "ref_elv"],
                 met_vars=['u', 'v', 'w'], emission_vars=['q_CH4']):

        self.min_trace_sensors = min_trace_sensors
        self.max_trace_sensors = max_trace_sensors
        self.min_leak_loc = min_leak_loc
        self.max_leak_loc = max_leak_loc
        self.sensor_height_min = sensor_height_min
        self.sensor_height_max = sensor_height_max
        self.leak_height_min = leak_height_min
        self.leak_height_max = leak_height_max
        self.sensor_exist_mask = sensor_exist_mask
        self.coord_vars = coord_vars
        self.met_vars = met_vars
        self.emission_vars = emission_vars
        self.variables = coord_vars + met_vars + emission_vars
        self.n_new_vars = 6
        self.met_loc_mask = np.isin(self.variables, self.emission_vars) * sensor_type_mask
        self.ch4_mask = np.isin(self.variables, self.met_vars) * sensor_type_mask

    def load_data(self, file_names, use_dask=True, swap_time_dim=True):

        """ load xarray datasets from a list of file names. """
        if swap_time_dim == True:
            self.data = xr.open_mfdataset(file_names, parallel=use_dask).swap_dims({'time': 'timeDim'}).load()
        else:
            self.data = xr.open_mfdataset(file_names, parallel=use_dask).load()
        
        self.time_steps = len(self.data['timeDim'].values)
        self.iDim = len(self.data.iDim)
        self.jDim = len(self.data.jDim)
        self.kDim = len(self.data.kDim) # needed for vert sampling
        self.x = self.data['xPos'][0, 0, :].values
        self.y = self.data['yPos'][0, :, 0].values
        self.z = self.data['zPos'][:, 0, 0].values
        self.z_res = self.data['zPos'][1, 0, 0].values - self.data['zPos'][0, 0, 0].values
        self.leak_rate = self.data['srcAuxScMassSpecValue']
        # add zero arrays for new derived variables
        for var in self.coord_vars:
            self.data[var] = (["kDim", "jDim", "iDim"], np.zeros(shape=(len(self.data.kDim),
                                                                        len(self.data.jDim),
                                                                        len(self.data.iDim))))

    def sample(self, time_window_size, samples_per_window, window_stride=5):

        """  Sample different geometric configurations of sensors from LES data for ML ingestion.
        Args:
            time_window_size (int): Length of timeseries (index based) for each sample.
            samples_per_window (int): Number of samples to draw from each time window.
            window_stride (int): stride width to slide window through time (index based)
        Returns:
            sensor_array, potential_leak_array: Numpy Arrays of shape (sample, sensor, time, variable) """

        sensor_arrays, leak_arrays, true_leak_idx = [], [], []
        step_size = np.arange(1, self.time_steps - time_window_size, window_stride)
        sensor_meta = np.zeros(shape=(samples_per_window * len(step_size), self.max_trace_sensors, 3))
        leak_meta = np.zeros(shape=(samples_per_window * len(step_size), self.max_leak_loc, 3))

        for i, t in enumerate(step_size):
            print(t)
            for s in range(samples_per_window):

                n_sensors = np.random.randint(low=self.min_trace_sensors, high=self.max_trace_sensors + 1)
                n_leaks = np.random.randint(low=self.min_leak_loc, high=self.max_leak_loc + 1)
                true_leak_pos = np.random.choice(n_leaks, size=1)[0]
                true_leak_i, true_leak_j = 15, 15

                # Sensor in ijk (xyz) space
                # X, Y samples the entire domain, and already in index space
                i_sensor = np.random.randint(low=0, high=self.iDim, size=n_sensors)
                j_sensor = np.random.randint(low=0, high=self.jDim, size=n_sensors)
                
                # Converting to index space
                senor_height_max_index = int(np.rint(self.sensor_height_max/self.z_res))
                senor_height_min_index = int(np.rint(self.sensor_height_min/self.z_res))

                ## Sensor vertical logic 
                if senor_height_max_index > self.kDim:
                    raise ValueError("Max sensor height is greater than domain, please pick a smaller number")
                elif self.sensor_height_min > senor_height_max_index:
                    raise ValueError("Min sensor height is greater than the maximum (in index space), please try again")
                elif senor_height_min_index == senor_height_max_index:
                    k_sensor = np.repeat(senor_height_max_index, 
                                         n_sensors)
                else:
                    k_sensor = np.random.randint(low=senor_height_min_index, 
                                                 high=senor_height_max_index, 
                                                 size=n_sensors)
                # end of sensor vertical sampling logic

                # Leaks in ijk (xyz) space
                i_leak = np.random.randint(low=0, high=self.iDim, size=n_leaks)
                j_leak = np.random.randint(low=0, high=self.jDim, size=n_leaks)

                # Converting to index space
                leak_height_max_index = int(np.rint(self.leak_height_max/self.z_res))
                leak_height_min_index = int(np.rint(self.leak_height_min/self.z_res))

                ## start of leak vertical logic 
                if leak_height_max_index > self.kDim:
                    raise ValueError("Max leak height is greater than domain, please pick a smaller number")
                elif self.leak_height_min > leak_height_max_index:
                    raise ValueError("Min leak height is greater than the maximum (in index space), please try again")
                elif leak_height_min_index == leak_height_max_index:
                    k_leak = np.repeat(leak_height_max_index, 
                                       n_leaks)
                else:
                    k_leak = np.random.randint(low=leak_height_min_index, 
                                               high=leak_height_max_index, 
                                               size=n_leaks)
                # end of vertical sample logic for leaks
                
                i_leak[true_leak_pos] = true_leak_i  # set one of the potential leaks to the true position
                j_leak[true_leak_pos] = true_leak_j

                sensor_phi = self.data[['w', 'v', 'u']].to_array().values[:, :,
                             k_sensor[0], j_sensor[0], i_sensor[0]][:, t:t + time_window_size].T
                sensor_array = np.zeros(shape=(6, n_sensors, time_window_size))
                for n in range(n_sensors):

                    sensor_idx = np.array([self.x[i_sensor[n]],
                                           self.y[j_sensor[n]],
                                           self.z[k_sensor[n]]])
                    sensor_meta[(i * samples_per_window) + s, n, :3] = sensor_idx
                    derived_vars = get_relative_azimuth(v=sensor_phi[:, 1],
                                                        u=sensor_phi[:, 2],
                                                        x_ref=self.x[i_sensor[0]],
                                                        y_ref=self.y[j_sensor[0]],
                                                        z_ref=self.z[k_sensor[0]],
                                                        x_target=self.x[i_sensor[n]],
                                                        y_target=self.y[j_sensor[n]],
                                                        z_target=self.z[k_sensor[n]],
                                                        time_series=True)
                    sensor_array[:, n, :] = derived_vars

                leak_array = np.zeros(shape=(6, n_leaks, 1))
                for l in range(n_leaks):

                    leak_idx = np.array([self.x[i_leak[l]],
                                         self.y[j_leak[l]],
                                         self.z[k_leak[l]]])
                    leak_meta[(i * samples_per_window) + s, l, :3] = leak_idx
                    derived_vars = get_relative_azimuth(v=sensor_phi[:, 1],
                                                        u=sensor_phi[:, 2],
                                                        x_ref=self.x[i_sensor[0]],
                                                        y_ref=self.y[j_sensor[0]],
                                                        z_ref=self.z[k_sensor[0]],
                                                        x_target=self.x[i_leak[l]],
                                                        y_target=self.y[j_leak[l]],
                                                        z_target=self.z[k_leak[l]],
                                                        time_series=False)
                    leak_array[:, l, :] = derived_vars
                sensor_sample = self.data[self.variables].to_array().expand_dims('sample').values[:, :,
                                self.sensor_height_min, self.sensor_height_max, j_sensor, i_sensor, t:t + time_window_size]
                leak_sample = self.data[self.variables].to_array().expand_dims('sample').values[:, :,
                                self.leak_height_min, self.leak_height_max, j_leak, i_leak, t:t+1]

                sensor_sample[0, :self.n_new_vars, :] = sensor_array
                sensor_sample = self.create_mask(sensor_sample, kind="sensor")
                leak_sample[0, :self.n_new_vars, :] = leak_array

                leak_sample = self.create_mask(leak_sample, kind="leak")
                padded_sensor_sample = self.pad_along_axis(sensor_sample, target_length=self.max_trace_sensors,
                                                           pad_value=self.sensor_exist_mask, axis=2)
                padded_leak_sample = self.pad_along_axis(leak_sample, target_length=self.max_leak_loc,
                                                         pad_value=self.sensor_exist_mask, axis=2)

                sensor_arrays.append(padded_sensor_sample)
                leak_arrays.append(padded_leak_sample)
                true_leak_idx.append(true_leak_pos)

        sensor_samples = np.transpose(np.vstack(sensor_arrays), axes=[0, 2, 1, 3, 4]) # order [samp, sensor, time, var]
        leak_samples = np.transpose(np.vstack(leak_arrays), axes=[0, 2, 1, 3, 4])
        targets = self.create_targets(leak_samples, true_leak_idx)

        return self.make_xr_ds(sensor_samples, leak_samples, targets, sensor_meta, leak_meta)

    def pad_along_axis(self, array, target_length, pad_value=0, axis=0):
        """ Pad numpy array along a single dimension. """
        pad_size = target_length - array.shape[axis]
        if pad_size <= 0:
            return array

        n_pad = [(0, 0)] * array.ndim
        n_pad[axis] = (0, pad_size)

        return np.pad(array, pad_width=n_pad, mode='constant', constant_values=pad_value)

    def create_mask(self, array, kind):

        array = np.transpose(array, axes=[0, 3, 2, 1])  # reshape for proper broadcasting
        mask_2d = np.zeros(shape=(array.shape[-2], array.shape[-1]))

        if kind == "sensor":

            mask_2d[0] = self.met_loc_mask  # make the first random sensor the "met sensor"
            mask_2d[1:] = self.ch4_mask     # all others don't have met data

        elif kind == "leak":

            mask_2d[:] = self.ch4_mask

        expanded_mask = np.broadcast_to(mask_2d, array.shape)
        array_w_mask = np.stack(arrays=[array, expanded_mask], axis=-1)

        return np.transpose(array_w_mask, axes=[0, 1, 2, 3, 4])

    def create_targets(self, leak_samples, true_leak_indices, categorical=True):
        """ Create target data from potential leak arrays. Outputs both concentrations and categorical (argmax)"""
        if categorical:
            targets = np.zeros(shape=(leak_samples.shape[0], leak_samples.shape[1]))
            np.put_along_axis(targets, np.array(true_leak_indices).reshape(-1, 1), 1, axis=1)
        else:
            targets = leak_samples[..., 0, -1, 0]

        return np.expand_dims(targets, axis=-1)

    def make_xr_ds(self, encoder_x, decoder_x, targets, sensor_meta, leak_meta):
        """ Convert numpy arrays from .sample() to xarray Arrays. """
        encoder_ds = xr.DataArray(encoder_x,
                                  dims=['sample', 'sensor', 'time', 'variable', 'mask'],
                                  coords={'variable': ["ref_distance", "ref_azi_sin", "ref_azi_cos", "ref_elv",
                                          "u", "v", "w", "q_CH4"]},
                                  name="encoder_input").astype('float32')

        decoder_ds = xr.DataArray(decoder_x,

                                  dims=['sample', 'pot_leak', 'target_time', 'variable', 'mask'],
                                  coords={'variable': ["ref_distance", "ref_azi_sin", "ref_azi_cos", "ref_elv",
                                          "u", "v", "w", "q_CH4"]},
                                  name="decoder_input").astype('float32')

        targets = xr.DataArray(targets,
                               dims=["sample", "pot_leak", "target_time"],
                               name="target").astype('int')

        target_ch4 = xr.DataArray(decoder_ds.sel(variable='q_CH4').isel(mask=0),
                                  name="target_ch4")

        sensor_locs = xr.DataArray(self.pad_along_axis(sensor_meta, target_length=self.max_trace_sensors,
                                                       pad_value=0, axis=1),
                                   dims=['sample', 'sensor', 'sensor_loc'],
                                   coords={'sensor_loc': ['xPos', 'yPos', 'zPos']},
                                   name="sensor_meta").astype('float32')

        leak_locs = xr.DataArray(self.pad_along_axis(leak_meta, target_length=self.max_leak_loc,
                                                     pad_value=0, axis=1),
                                 dims=['sample', 'pot_leak', 'sensor_loc'],
                                 coords={'sensor_loc': ['xPos', 'yPos', 'zPos']},
                                 name="leak_meta").astype('float32')

        met_sensor_loc = xr.DataArray(sensor_locs[:, 0],
                                      dims=['sample', 'sensor_loc'],
                                      coords={'sensor_loc': ['xPos', 'yPos', 'zPos']},
                                      name="met_sensor_loc").astype('float32')

        leak_rate = xr.DataArray(np.repeat(self.leak_rate, targets.shape[0]).astype('float32'),
                                 dims=['sample'],
                                 name='leak_rate')

        ds = xr.merge([encoder_ds, decoder_ds, targets, target_ch4, sensor_locs, leak_locs, met_sensor_loc, leak_rate])
        return ds


class Preprocessor():

    def __init__(self, scaler_type="quantile", sensor_pad_value=None, sensor_type_value=None):

        self.sensor_pad_value = sensor_pad_value
        self.sensor_type_value = sensor_type_value

        if scaler_type.lower() == "standard":
            self.scaler = DeepStandardScaler()
        elif scaler_type.lower() == "minmax":
            self.scaler = DeepMinMaxScaler()
        elif scaler_type.lower() == "quantile":
            self.scaler = DeepQuantileTransformer()

    def load_data(self, files):

        ds = xr.open_mfdataset(files, concat_dim='sample', combine="nested", parallel=False)
        encoder_data = ds['encoder_input']
        decoder_data = ds['decoder_input']
        leak_location = ds['target'].values
        leak_rate = ds['leak_rate'].values

        return encoder_data, decoder_data, leak_location.squeeze(), leak_rate

    def save_filenames(self, train_files, validation_files, out_path):
        if not exists(out_path):
            makedirs(out_path)
        train_file_series = pd.Series(train_files, name="train_files")
        train_file_series.to_csv(join(out_path, "train_files.csv"))
        validation_file_series = pd.Series(validation_files, name="validation_files")
        validation_file_series.to_csv(join(out_path, "validation_files.csv"))

    def preprocess(self, data, fit_scaler=True):

        imputed_data, mask = self.impute_mask(data)
        padding_mask = mask[..., 0, 0]

        if fit_scaler:
            self.fit_scaler(imputed_data)

        scaled_data = self.transform(imputed_data)
        scaled_data = self.inv_impute_mask(scaled_data, mask).squeeze()

        return scaled_data, ~padding_mask

    def impute_mask(self, data):

        arr = data[..., 0].values
        mask = data[..., -1].values

        arr[mask == self.sensor_pad_value] = np.nan
        arr[mask == self.sensor_type_value] = np.nan

        new_mask = np.zeros(shape=mask.shape)
        new_mask[mask == self.sensor_pad_value] = 1
        new_mask[mask == self.sensor_type_value] = 1

        return arr, new_mask.astype(bool)

    def inv_impute_mask(self, data, mask, impute_value=0):

        data[mask == True] = impute_value

        return data

    def fit_scaler(self, data):

        self.scaler.fit(data)

    def transform(self, data):

        scaled_data = self.scaler.transform(data)

        return scaled_data


def save_output(out_path, train_targets, val_targets, train_predictions, val_predictions, model_name):

    if model_name == "transformer_leak_loc" or model_name == "gaussian_process":

        train_output = xr.Dataset(data_vars=dict(target_pot_loc=(["sample", "pot_leak_locs"], train_targets),
                                                 leak_loc_pred=(["sample", "pot_leak_locs"], train_predictions)))
        val_output = xr.Dataset(data_vars=dict(targets=(["sample", "pot_leak_locs"], val_targets),
                                               leak_loc_pred=(["sample", "pot_leak_locs"], val_predictions)))

    elif model_name == "transformer_leak_rate":

        train_output = xr.Dataset(data_vars=dict(target_leak_rate=(["sample"], train_targets),
                                                 leak_rate_pred=(["sample"], train_predictions)))
        val_output = xr.Dataset(data_vars=dict(target_leak_rate=(["sample"], val_targets),
                                               leak_rate_pred=(["sample"], val_predictions)))

    elif model_name == "backtracker":
        train_output = xr.Dataset(data_vars=dict(target_pot_loc=(["sample", "pot_leak_locs"], train_targets),
                                                 target_leak_rate=(["sample"], train_predictions[:, -2]),
                                                 leak_loc_pred=(["sample", "pot_leak_locs"], train_predictions[:, -2]),
                                                 leak_loc_pred_coords=(["sample", "x", "y", "z"], train_predictions[:, :-1]),
                                                 leak_rate_pred=(["sample"], train_predictions[3])))
        val_output = xr.Dataset(data_vars=dict(target_pot_loc=(["sample", "pot_leak_locs"], val_targets),
                                                 target_leak_rate=(["sample"], val_predictions[:, -2]),
                                                 leak_loc_pred=(["sample", "pot_leak_locs"], val_predictions[:, -2]),
                                                 leak_loc_pred_coords=(["sample", "x", "y", "z"], val_predictions[:, :-1]),
                                                 leak_rate_pred=(["sample"], val_predictions[3])))
    else:
        raise ValueError(f"Model name {model_name} not found.")

    train_output.to_netcdf(out_path)
    val_output.to_netcdf(out_path)

    return
