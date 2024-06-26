import xarray as xr
import pandas as pd
import numpy as np
from os.path import join, exists
from os import makedirs
from scipy.ndimage import minimum_filter
from sealsml.geometry import GeoCalculator, get_relative_azimuth, generate_sensor_positions_min_distance
from bridgescaler import DQuantileScaler, DMinMaxScaler, DStandardScaler, load_scaler, save_scaler


class DataSampler(object):
    """ Sample LES data with various geometric configurations. """

    def __init__(self, min_trace_sensors=3, max_trace_sensors=15, min_leak_loc=1, max_leak_loc=10,
                 sensor_height_min=1,
                 sensor_height_max=4,
                 leak_height_min=0,
                 leak_height_max=4,
                 sensor_type_mask=1, 
                 sensor_exist_mask=-1,
                 sensor_min_distance=20.0,
                 coord_vars=None,
                 met_vars=None, emission_vars=None,
                 pot_leaks_scheme=None, pot_leaks_file=None, sensor_sampling_strategy=None):
        if coord_vars is None:
            coord_vars = ["ref_distance", "ref_azi_sin", "ref_azi_cos", "ref_elv"]
        if met_vars is None:
            met_vars = ['u', 'v', 'w']
        if emission_vars is None:
            emission_vars = ['q_CH4']
        if pot_leaks_scheme == None:
            pot_leaks_scheme = 'random_sampling'
        if pot_leaks_file == None:
            pot_leaks_file = ''
        if sensor_sampling_strategy == None:
            sensor_sampling_strategy = 'random_sampling'
        self.min_trace_sensors = min_trace_sensors
        self.max_trace_sensors = max_trace_sensors
        self.max_met_sensors = 1  # Currently self.max_met_sensors strictly permitted to be 1
        self.min_leak_loc = min_leak_loc
        self.max_leak_loc = max_leak_loc
        self.sensor_height_min = sensor_height_min
        self.sensor_height_max = sensor_height_max
        self.sensor_sampling_strategy = sensor_sampling_strategy
        self.sensor_min_distance = sensor_min_distance
        self.leak_height_min = leak_height_min
        self.leak_height_max = leak_height_max
        self.sensor_exist_mask = sensor_exist_mask
        self.coord_vars = coord_vars
        self.met_vars = met_vars
        self.emission_vars = emission_vars
        self.variables = coord_vars + met_vars + emission_vars
        # Total number of coord + wind variables after mean-wind-relative grid rotation.
        # Only u and v are changed, so w is ignored.
        self.n_rotated_vars = len(coord_vars) + len(met_vars[:2])
        self.met_loc_mask = np.isin(self.variables, self.emission_vars) * sensor_type_mask
        self.ch4_mask = np.isin(self.variables, self.met_vars) * sensor_type_mask
        self.leak_mask = np.isin(self.variables, self.met_vars + self.emission_vars) * sensor_type_mask
        self.pot_leaks_scheme = pot_leaks_scheme
        self.pot_leaks_file = pot_leaks_file

    def load_data(self, file_names, use_dask=True, swap_time_dim=True):
        '''Dataset loader that exports an xarray ds and '''
        if swap_time_dim == True:
            ds = xr.open_mfdataset(file_names, parallel=use_dask).swap_dims({'time': 'timeDim'})
        else:
            ds = xr.open_mfdataset(file_names, parallel=use_dask)
        # need the number of sources
        num_sources = ds.sizes['srcDim']
        # may need the structureMask and/or shell_mask and self.max_leak_loc to be set accordingly
        if 'structureMask' in ds.variables:
            if self.pot_leaks_scheme == 'full_mask':
                # set the max and min leak locs to be the number of structure mask cells + 1 (possibly disjoint) true leak
                self.max_leak_loc = np.argwhere(ds['structureMask'].values > 0).shape[0] + 1  # padded by 1 for a true leak
                self.min_leak_loc = self.max_leak_loc
            elif self.pot_leaks_scheme == 'shell_mask':
                self.setShellMask(ds)
                # set the max and min leak locs to be the number of shell_mask cells + 1 (possibly disjoint) true leak
                self.max_leak_loc = np.argwhere(self.shell_mask > 0).shape[0] + 1  # padded by 1 for a true leak
                self.min_leak_loc = self.max_leak_loc
            elif self.pot_leaks_scheme == 'from_pot_leak_file':
                self.ds_pot_leaks = xr.open_dataset(self.pot_leaks_file)  # lazy evaluation is likely fine here
                self.max_leak_loc = self.ds_pot_leaks.sizes['plDim'] + 1  # padded by 1 for a true leak
                self.min_leak_loc = self.max_leak_loc
            else:
                self.ds_pot_leaks = None
                self.shell_mask = np.asarray([])
        else:
            print("No structureMask in the input dataset, structureMask is an expected DataArray field of the input dataset.")

        print(f"pot_leaks_scheme = {self.pot_leaks_scheme}, max_leak_loc = {self.max_leak_loc}, min_leak_loc = {self.min_leak_loc}")
        return ds, num_sources

    def data_extract(self, ds):

        if not isinstance(ds, (xr.Dataset, xr.DataArray)):
            print("Error: The provided input is not an xarray Dataset or DataArray.")
            return

        self.data = ds.load()
        self.time_steps = len(self.data['timeDim'].values)
        self.iDim = self.data.sizes['iDim']
        self.jDim = self.data.sizes['jDim']
        self.kDim = self.data.sizes['kDim']  # needed for vert sampling
        self.x = self.data['xPos'][0, 0, :].values
        self.y = self.data['yPos'][0, :, 0].values
        self.z = self.data['zPos'][:, 0, 0].values
        self.z_res = self.data['zPos'][1, 0, 0].values - self.data['zPos'][0, 0, 0].values
        self.x_res = self.data['xPos'][1, 0, 0].values - self.data['xPos'][0, 0, 0].values
        self.leak_rate = self.data['srcAuxScMassSpecValue'].values
        self.leak_loc = self.data['srcAuxScLocation'].values

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
        # Create an iterable vector of start time indices corresponding to each targeted window of the full time series
        time_window_starts = np.arange(0, self.time_steps - time_window_size, window_stride)
        num_windows = len(time_window_starts)  # number of windows in the full time series
        sensor_meta = np.zeros(shape=(samples_per_window * num_windows,
                                      self.max_trace_sensors + self.max_met_sensors, 3))
        # Currently self.max_met_sensors strictly enforced to be 1
        leak_meta = np.zeros(shape=(samples_per_window * num_windows, self.max_leak_loc, 3))
        mean_wd = np.zeros(samples_per_window * num_windows)
        for i, t in enumerate(time_window_starts):
            print(t)
            for s in range(samples_per_window):

                # Set the sensor locations and time-series data streams within this sample
                # Total number of sensors (currently 1 met-sensor + random sample of trace sensors in specified range)
                n_sensors = np.random.randint(low=self.min_trace_sensors,
                                              high=self.max_trace_sensors + 1) + self.max_met_sensors

                # Sensor in ijk (xyz) space
                # X, Y samples the entire domain, and already in index space
                if self.sensor_sampling_strategy == 'random_sampling':
                    i_sensor = np.random.randint(low=0, high=self.iDim, size=n_sensors)
                    j_sensor = np.random.randint(low=0, high=self.jDim, size=n_sensors)
                elif self.sensor_sampling_strategy == 'minimum_distance':
                    # This does not take into account vertical componet
                    i_sensor, j_sensor = generate_sensor_positions_min_distance(n_sensors,
                                                                                self.x,
                                                                                self.y,
                                                                                min_distance=self.sensor_min_distance)
                else:
                    print('bad strategy')

                # Converting to index space
                sensor_height_max_index = int(np.rint(self.sensor_height_max / self.z_res))
                sensor_height_min_index = int(np.rint(self.sensor_height_min / self.z_res))

                ## Sensor vertical logic
                if sensor_height_max_index > self.kDim:
                    raise ValueError("Max sensor height is greater than domain, please pick a smaller number")
                elif sensor_height_min_index > sensor_height_max_index:
                    raise ValueError("Min sensor height is greater than the maximum (in index space), please try again")
                elif sensor_height_min_index == sensor_height_max_index:
                    k_sensor = np.repeat(sensor_height_max_index,
                                         n_sensors)
                else:
                    k_sensor = np.random.randint(low=sensor_height_min_index,
                                                 high=sensor_height_max_index,
                                                 size=n_sensors)
                # end of sensor vertical sampling logic

                sensor_phi = self.data[self.met_vars].isel({'timeDim': slice(t, t + time_window_size),
                                                            'kDim': k_sensor[0],
                                                            'jDim': j_sensor[0],
                                                            'iDim': i_sensor[0]}).to_array().values.T
                sensor_sample = np.zeros(shape=(1, len(self.variables), n_sensors, time_window_size))
                for n in range(n_sensors):
                    sensor_idx = np.array([self.x[i_sensor[n]],
                                           self.y[j_sensor[n]],
                                           self.z[k_sensor[n]]])
                    sensor_meta[(i * samples_per_window) + s, n, :3] = sensor_idx
                    derived_vars, tmp_wd = get_relative_azimuth(u=sensor_phi[:, 0],
                                                                v=sensor_phi[:, 1],
                                                                x_ref=self.x[i_sensor[0]],
                                                                y_ref=self.y[j_sensor[0]],
                                                                z_ref=self.z[k_sensor[0]],
                                                                x_target=self.x[i_sensor[n]],
                                                                y_target=self.y[j_sensor[n]],
                                                                z_target=self.z[k_sensor[n]],
                                                                time_series=True)
                    sensor_sample[0, 0:derived_vars.shape[0], n,
                    :] = derived_vars  # sensor-n: coord_vars + met: u_rot,v_rot
                    sensor_sample[0, -2, n, :] = sensor_phi[:, 2].T  # met: w
                    sensor_sample[0, -1, n, :] = self.data[self.emission_vars[0]][t:t + time_window_size:, k_sensor[n],
                                                 j_sensor[n], i_sensor[n]].values.T  # sensor-n: emission_vars
                mean_wd[(i * samples_per_window) + s] = tmp_wd

                sensor_sample_masked = self.create_mask(sensor_sample, kind="sensor")
                padded_sensor_sample = self.pad_along_axis(sensor_sample_masked,
                                                           target_length=self.max_trace_sensors + self.max_met_sensors,
                                                           pad_value=self.sensor_exist_mask, axis=2)
                sensor_arrays.append(padded_sensor_sample)

                # Set potential leak locations within this sample
                if self.pot_leaks_scheme == 'full_mask':
                    # Single call to set the potential leak locations as the structure mask along with the true leak
                    n_leaks, true_leak_pos, i_leak, j_leak, k_leak = self.setupStructureMaskLeakLocations(
                        self.data['structureMask'].values)
                elif self.pot_leaks_scheme == 'shell_mask':
                    # Single call to set the potential leak locations as the shell mask along with the true leak
                    n_leaks, true_leak_pos, i_leak, j_leak, k_leak = self.setupStructureMaskLeakLocations(
                        self.shell_mask)
                elif self.pot_leaks_scheme == 'from_pot_leak_file':
                    # Single call to set the potential leak locations as specified from a NetCDF file along with the true leak
                    n_leaks, true_leak_pos, i_leak, j_leak, k_leak = self.setupSpecifiedLeakLocations()
                else:
                    # self.pot_leaks_scheme == 'random_sampling':
                    # Single call to randomly set the potential leak locations and a random-indexed true leak
                    n_leaks, true_leak_pos, i_leak, j_leak, k_leak = self.setupRandomLeakLocations()

                # Save the pot_leak index of the randomly-indexed true leak
                true_leak_idx.append(true_leak_pos)

                # Map the potential leak locations into mean wind realtive coordinate frame
                leak_sample = np.zeros(shape=(1, len(self.variables), n_leaks, 1))
                for l in range(n_leaks):
                    leak_idx = np.array([self.x[i_leak[l]],
                                         self.y[j_leak[l]],
                                         self.z[k_leak[l]]])
                    leak_meta[(i * samples_per_window) + s, l, :3] = leak_idx
                    derived_vars, tmp_wd = get_relative_azimuth(u=sensor_phi[:, 0],
                                                                v=sensor_phi[:, 1],
                                                                x_ref=self.x[i_sensor[0]],
                                                                y_ref=self.y[j_sensor[0]],
                                                                z_ref=self.z[k_sensor[0]],
                                                                x_target=self.x[i_leak[l]],
                                                                y_target=self.y[j_leak[l]],
                                                                z_target=self.z[k_leak[l]],
                                                                time_series=False)
                    # Set the wind-relative coordinate variables for this pot_leak, leave the met_vars+emission_vars to be imputed during preprocessing
                    leak_sample[0, 0:len(self.coord_vars), l, :] = derived_vars[0:len(self.coord_vars), :]

                leak_sample_masked = self.create_mask(leak_sample, kind="leak")
                padded_leak_sample = self.pad_along_axis(leak_sample_masked, target_length=self.max_leak_loc,
                                                         pad_value=self.sensor_exist_mask, axis=2)

                leak_arrays.append(padded_leak_sample)

        # Finalize shapes of sensor (encoder) and leak (decoder) sampled data arrays
        sensor_samples = np.transpose(np.vstack(sensor_arrays), axes=[0, 2, 1, 3, 4])  # order [samp, sensor, time, var, mask]
        leak_samples = np.transpose(np.vstack(leak_arrays), axes=[0, 2, 1, 3, 4])
        targets = self.create_targets(leak_samples, true_leak_idx)

        return self.make_xr_ds(sensor_samples, leak_samples, targets, sensor_meta, leak_meta, mean_wd)

    def setupRandomLeakLocations(self):
        # Number of potential leak locations
        n_leaks = np.random.randint(low=self.min_leak_loc, high=self.max_leak_loc + 1)

        # Leaks in ijk (xyz) space
        i_leak = np.random.randint(low=0, high=self.iDim, size=n_leaks)
        j_leak = np.random.randint(low=0, high=self.jDim, size=n_leaks)

        # Converting to index space
        leak_height_max_index = int(np.rint(self.leak_height_max / self.z_res))
        leak_height_min_index = int(np.rint(self.leak_height_min / self.z_res))

        ## start of leak vertical logic
        if leak_height_max_index > self.kDim:
            raise ValueError("Max leak height is greater than domain, please pick a smaller number")
        elif self.leak_height_min > leak_height_max_index:
            raise ValueError("Min leak height is greater than the maximum (in index space), please try again")
        elif leak_height_min_index == leak_height_max_index:
            k_leak = np.repeat(leak_height_max_index, n_leaks)
        else:
            k_leak = np.random.randint(low=leak_height_min_index,
                                       high=leak_height_max_index,
                                       size=n_leaks)
        # end of vertical sample logic for leaks

        ### Set the true leak
        # Randomize the index of the true leak within the set of randomly located potential leaks
        true_leak_pos = np.random.choice(n_leaks, size=1)[0]
        # Find the true leak indices
        true_leak_i, true_leak_j, true_leak_k = self.findIndices(self.leak_loc[0], self.leak_loc[1], self.leak_loc[2])
        # set the random-indexed potential leak to the true leak position
        i_leak[true_leak_pos] = true_leak_i
        j_leak[true_leak_pos] = true_leak_j
        k_leak[true_leak_pos] = true_leak_k

        return n_leaks, true_leak_pos, i_leak, j_leak, k_leak

    def setupSpecifiedLeakLocations(self, randomOrdering=True):
        n_leaks = self.ds_pot_leaks.sizes['plDim']
        pl_indices = np.zeros((n_leaks, 3), dtype=np.int32)

        for idx in range(n_leaks):
            pl_indices[idx, 0], pl_indices[idx, 1], pl_indices[idx, 2] = self.findIndices(
                self.ds_pot_leaks['srcPotLeakLocation'][idx, 0].values,
                self.ds_pot_leaks['srcPotLeakLocation'][idx, 1].values,
                self.ds_pot_leaks['srcPotLeakLocation'][idx, 2].values)
        if randomOrdering:
            # randomize the order of the potential leaks
            np.random.shuffle(pl_indices)  # only randomizes the first dimension (rows)
        ### Set the true leak
        # Find the true leak indices
        # true_leak_i, true_leak_j, true_leak_k = self.findTrueLeakIndices()
        true_leak_i, true_leak_j, true_leak_k = self.findIndices(self.leak_loc[0], self.leak_loc[1], self.leak_loc[2])
        # Look for the true leak in the potential leaks
        indx = np.squeeze(np.argwhere(np.all((pl_indices - [true_leak_i, true_leak_j, true_leak_k] == 0), axis=-1)))
        if indx.size > 0:  # the true leak location is already in the randomly ordered potential leaks
            true_leak_pos = indx  # set the return value for the index of the true leak
        else:  # The true location isn't already in the potential leaks
            # Randomize the index of the true leak within the set of randomly located potential leaks
            true_leak_pos = np.random.choice(n_leaks, size=1)[0]
            tmp_indices = pl_indices
            pl_indices = np.append(tmp_indices, np.expand_dims(tmp_indices[true_leak_pos, :], axis=0), axis=0)
            pl_indices[true_leak_pos, :] = [true_leak_i, true_leak_j, true_leak_k]
            n_leaks += 1  # Increment the number of pot_leaks by 1 to account for the added true leak
        i_leak = pl_indices[:, 0]
        j_leak = pl_indices[:, 1]
        k_leak = pl_indices[:, 2]

        return n_leaks, true_leak_pos, i_leak, j_leak, k_leak

    def setupStructureMaskLeakLocations(self, structureMask, randomOrdering=True):
        # Find all the indices where structure mask is nonzero
        structmask_indices = np.argwhere(structureMask > 0)[:, -1::-1]  # reverse the structureMask k,j,i ordering to use i,j,k
        # Assume every structure masked cell is a potential leak
        n_leaks = structmask_indices.shape[0]
        if randomOrdering:
            # randomize the order of the potential leaks
            np.random.shuffle(structmask_indices)  # only randomizes the first dimension (rows)
        ### Set the true leak
        # Find the true leak indices
        true_leak_i, true_leak_j, true_leak_k = self.findIndices(self.leak_loc[0], self.leak_loc[1], self.leak_loc[2])
        # Look for the true leak in the potential leaks
        indx = np.squeeze(np.argwhere(np.all((structmask_indices - [true_leak_i, true_leak_j, true_leak_k] == 0), axis=-1)))
        if indx.size > 0:  # the true leak location is already in the randomly ordered potential leaks
            true_leak_pos = indx  # set the return value for the index of the true leak
        else:  # The true location isn't already in the potential leaks
            # Randomize the index of the true leak within the set of randomly located potential leaks
            true_leak_pos = np.random.choice(n_leaks, size=1)[0]
            tmp_indices = structmask_indices
            structmask_indices = np.append(tmp_indices, np.expand_dims(tmp_indices[true_leak_pos, :], axis=0), axis=0)
            structmask_indices[true_leak_pos, :] = [true_leak_i, true_leak_j, true_leak_k]
            n_leaks += 1  # Increment the number of pot_leaks by 1 to account for the added true leak
        i_leak = structmask_indices[:, 0]
        j_leak = structmask_indices[:, 1]
        k_leak = structmask_indices[:, 2]

        return n_leaks, true_leak_pos, i_leak, j_leak, k_leak

    def setShellMask(self, ds):
        full_mask = ds['structureMask'].values
        # Identify (with a value of 1 ) any structure mask cells bounded on all sides
        # (bottom boundary is reflected) by other structure mask cells
        interior_cells = minimum_filter(full_mask, size=(3, 3, 3))
        # Subtract interior cells from the full mask to mask only cells with one or more open-air neighbors
        shell_mask = full_mask - interior_cells
        self.shell_mask = shell_mask
        return

    def findIndices(self, xloc, yloc, zloc):

        i_indx = np.abs(self.x - xloc).argmin()
        j_indx = np.abs(self.y - yloc).argmin()
        k_indx = np.abs(self.z - zloc).argmin()

        return i_indx, j_indx, k_indx

    def pad_along_axis(self, array, target_length, pad_value=0, axis=0):
        """ Pad numpy array along a single dimension. """
        pad_size = target_length - array.shape[axis]
        if pad_size <= 0:
            return array

        n_pad = [(0, 0)] * array.ndim
        n_pad[axis] = (0, pad_size)

        return np.pad(array, pad_width=n_pad, mode='constant', constant_values=pad_value)

    def create_mask(self, array, kind):

        array = np.transpose(array, axes=[0, 3, 2, 1])  # reshape for proper broadcasting assuming array was [samp,var,sensor,time]
        mask_2d = np.zeros(shape=(array.shape[-2], array.shape[-1]))

        if kind == "sensor":

            mask_2d[0] = self.met_loc_mask  # make the first random sensor the "met sensor"
            mask_2d[1:] = self.ch4_mask  # all others don't have met data

        elif kind == "leak":

            mask_2d[:] = self.leak_mask

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

    def make_xr_ds(self, encoder_x, decoder_x, targets, sensor_meta, leak_meta, mn_wd):
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

        mean_wd = xr.DataArray(mn_wd,
                               dims=['sample'],
                               name='mean_wd')

        ds = xr.merge([encoder_ds, decoder_ds, targets, sensor_locs, leak_locs, met_sensor_loc, leak_rate, mean_wd])
        return ds


class Preprocessor():

    def __init__(self, scaler_type="quantile", sensor_pad_value=None, sensor_type_value=None):

        self.sensor_pad_value = sensor_pad_value
        self.sensor_type_value = sensor_type_value

        if scaler_type.lower() == "standard":
            self.coord_scaler = DStandardScaler()
            self.sensor_scaler = DStandardScaler()
        elif scaler_type.lower() == "minmax":
            self.coord_scaler = DMinMaxScaler()
            self.sensor_scaler = DMinMaxScaler()
        elif scaler_type.lower() == "quantile":
            self.coord_scaler = DQuantileScaler()
            self.sensor_scaler = DQuantileScaler()

    def load_data(self, files):

        ds = xr.open_mfdataset(files, concat_dim='sample', combine="nested", parallel=False, engine='netcdf4')
        encoder_data = ds['encoder_input'].load()
        decoder_data = ds['decoder_input'].load()
        leak_location = ds['target'].values
        leak_rate = ds['leak_rate'].values

        return encoder_data, decoder_data, leak_location.squeeze(), leak_rate

    def load_scalers(self, coord_scaler_path, sensor_scaler_path):

        self.coord_scaler = load_scaler(coord_scaler_path)
        self.sensor_scaler = load_scaler(sensor_scaler_path)

    def save_scalers(self, out_path):

        save_scaler(self.coord_scaler, join(out_path, f"coord_scaler.json"))
        save_scaler(self.sensor_scaler, join(out_path, f"sensor_scaler.json"))


    def save_filenames(self, train_files, validation_files, out_path):

        if not exists(out_path):
            makedirs(out_path)
        train_file_series = pd.Series(train_files, name="train_files")
        train_file_series.to_csv(join(out_path, "train_files.csv"))
        validation_file_series = pd.Series(validation_files, name="validation_files")
        validation_file_series.to_csv(join(out_path, "validation_files.csv"))

    def preprocess(self, encoder_data, decoder_data, fit_scaler=False):

        imputed_encoder_data, encoder_mask = self.impute_mask(encoder_data)
        encoder_padding_mask = encoder_mask[..., 0, 0]
        imputed_decoder_data, decoder_mask = self.impute_mask(decoder_data)
        decoder_padding_mask = decoder_mask[..., 0, 0]
        if fit_scaler:
            self.fit_scaler(imputed_encoder_data,imputed_decoder_data)
        scaled_encoder_data, scaled_decoder_data = self.transform(imputed_encoder_data, imputed_decoder_data)
        scaled_encoder_data = self.inv_impute_mask(scaled_encoder_data, encoder_mask).squeeze()
        scaled_decoder_data = self.inv_impute_mask(scaled_decoder_data, decoder_mask).squeeze()


        return scaled_encoder_data, scaled_decoder_data, ~encoder_padding_mask, ~decoder_padding_mask

    def impute_mask(self, data):
        arr = data[..., 0].values
        mask = data[..., -1].values

        arr[mask == self.sensor_pad_value] = np.nan
        arr[mask == self.sensor_type_value] = np.nan

        new_mask = np.zeros(shape=mask.shape)
        new_mask[mask == self.sensor_pad_value] = 1
        new_mask[mask == self.sensor_type_value] = 1

        return arr, new_mask.astype(bool)

    def inv_impute_mask(self, data, mask, impute_value=0, impute_wind_vals=True):

        data[mask == True] = impute_value

        if impute_wind_vals:

            winds = data[:, 0:1, :, 4:7]
            data[:, 1:, :, 4:7] = winds # impute wind data on all ch4 sensors

        return data

    def fit_scaler(self, encoder_data, decoder_data):

        # pull a single timestep from encoder to match decoder and concatenate along sensor / pot_leak_loc dim
        all_coord_data = np.concatenate([encoder_data[:, :, 0:1, :4], decoder_data[..., :4]], axis=1)

        self.coord_scaler.fit(all_coord_data)
        self.sensor_scaler.fit(encoder_data[..., -4:])


    def transform(self, encoder_data, decoder_data):

        scaled_encoder_coords = self.coord_scaler.transform(encoder_data[..., :4])
        scaled_encoder_sensors = self.sensor_scaler.transform((encoder_data[..., -4:]))
        scaled_encoder_data = np.concatenate([scaled_encoder_coords,scaled_encoder_sensors], axis=-1)

        scaled_decoder_coords = self.coord_scaler.transform(decoder_data[..., :4])
        scaled_decoder_data = np.concatenate([scaled_decoder_coords, decoder_data[..., -4:]], axis=-1)

        return scaled_encoder_data, scaled_decoder_data


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

    train_output.to_netcdf(out_path.split('.')[0] + "_train.nc")
    val_output.to_netcdf(out_path.split('.')[0] + "_val.nc")

    return
