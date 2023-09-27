import xarray as xr
import numpy as np

class DataSampler(object):
    """ Sample LES data with various geometric configurations. """

    def __init__(self, min_trace_sensors=3, max_trace_sensors=15, min_leak_loc=1, max_leak_loc=10,
                 sensor_height=3, resolution=2, coord_vars=["ref_distance", "ref_azi", "ref_elv"],
                 met_vars=['u', 'v', 'w'], emission_vars=['q_CH4']):

        self.min_trace_sensors = min_trace_sensors
        self.max_trace_sensors = max_trace_sensors
        self.min_leak_loc = min_leak_loc
        self.max_leak_loc = max_leak_loc
        self.sensor_height = sensor_height
        self.resolution = resolution
        self.coord_vars = coord_vars
        self.met_vars = met_vars
        self.emission_vars = emission_vars
        self.variables = coord_vars + met_vars + emission_vars

    def load_data(self, file_names):

        """ load xarray datasets from a list of file names. """

        self.data = xr.open_mfdataset(file_names, parallel=True).swap_dims({'time': 'timeDim'}).load()
        self.time_steps = len(self.data['timeDim'].values)
        self.iDim = len(self.data.iDim)
        self.jDim = len(self.data.jDim)
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

        sensor_arrays, leak_arrays = [], []

        for t in np.arange(0, self.time_steps - time_window_size, window_stride):
            print(t)
            for _ in range(samples_per_window):

                n_sensors = np.random.randint(low=self.min_trace_sensors, high=self.max_trace_sensors)
                n_leaks = np.random.randint(low=self.min_leak_loc, high=self.max_leak_loc)
                true_leak_pos = 0

                reference_point = np.random.randint(low=0, high=self.iDim, size=3)
                reference_point[-1] = self.sensor_height
                true_leak_i, true_leak_j = 15, 15

                i_sensor = np.random.randint(low=0, high=self.iDim, size=n_sensors)
                j_sensor = np.random.randint(low=0, high=self.jDim, size=n_sensors)
                i_leak = np.random.randint(low=0, high=self.iDim, size=n_leaks)
                j_leak = np.random.randint(low=0, high=self.jDim, size=n_leaks)
                i_leak[true_leak_pos] = true_leak_i  # set one of the potential leaks to the true position
                j_leak[true_leak_pos] = true_leak_j
                k = self.sensor_height

                sensor_sample = self.data[self.variables].to_array().expand_dims('sample').values[:, :,
                                k, j_sensor, i_sensor, t:t + time_window_size]
                leak_sample = self.data[self.variables].to_array().expand_dims('sample').values[:, :,
                                k, j_leak, i_leak, t + time_window_size: t + time_window_size + 1]

                derived_sensor_vars = self.derive_variables(i_sensor, j_sensor, np.repeat(k, n_sensors),
                                                            reference_point)
                derived_leak_vars = self.derive_variables(i_leak, j_leak, np.repeat(k, n_leaks), reference_point)

                expanded_vars = np.transpose(np.broadcast_to(derived_sensor_vars,
                                                             shape=(time_window_size, derived_sensor_vars.shape[0], 3)),
                                                             axes=[2, 1, 0])
                sensor_sample[0, :3, :] = expanded_vars
                leak_sample[0, :3, :, 0] = derived_leak_vars.T

                padded_sensor_sample = self.pad_along_axis(sensor_sample, target_length=self.max_trace_sensors,
                                                           pad_value=0, axis=2)
                padded_leak_sample = self.pad_along_axis(leak_sample, target_length=self.max_leak_loc,
                                                         pad_value=0, axis=2)

                sensor_arrays.append(padded_sensor_sample)
                leak_arrays.append(padded_leak_sample)

        sensor_samples = np.transpose(np.vstack(sensor_arrays), axes=[0, 2, 3, 1])
        leak_samples = np.transpose(np.vstack(leak_arrays), axes=[0, 2, 3, 1])

        return sensor_samples, leak_samples

    def derive_variables(self, i, j, k, reference_point):
        """ derive variables from randomly sampled reference point: Distance (in meters), azimuth angle,
         and elevation angle.
         Args:
             i (array): randomly sampled indices in i direction
             j (array): randomly sampled indices in j direction
             k (array): randomly sampled indices in k direction
             reference_point (array): Randomly sampled reference point (i, j, k)
        Returns:
             Stacked np.array of (distances, azimuths, elevations)
         """
        sample_points = np.stack([i, j, k]).T
        reference_points = np.broadcast_to(reference_point, shape=sample_points.shape)

        distances = self.calc_euclidean_dist(sample_points, reference_points)
        azimuths = self.calc_azimuth(sample_points, reference_points)
        elevations = self.calc_elevation(sample_points, reference_points)

        new_vars = np.stack([distances, azimuths, elevations]).T

        return new_vars

    def calc_euclidean_dist(self, sample_array, reference_array, axis=1):
        """ Calculate the Euclidean distance from the reference point (in meters) """
        return np.linalg.norm(sample_array - reference_array, axis=axis) * self.resolution

    def calc_azimuth(self, sample_array, reference_array):
        """ Calculate the azimuth angles from the reference point (degrees) """
        reference, points = reference_array[:, :-1], sample_array[:, :-1]  # remove k dimension
        diff = reference - points
        angle_degree = np.degrees(np.arctan2(diff[:, 1].astype('float32'), diff[:, 0].astype('float32')))

        return angle_degree

    def calc_elevation(self, sample_array, reference_array):
        """ Calculate the elevation angles from the reference point (degrees). """
        diff = reference_array - sample_array
        angle_degree = np.degrees(np.arctan2(diff[:, 2].astype('float32'), diff[:, 1].astype('float32')))

        return angle_degree

    def pad_along_axis(self, array, target_length, pad_value=0, axis=0):
        """ Pad numpy array along a single dimension. """
        pad_size = target_length - array.shape[axis]
        if pad_size <= 0:
            return array

        n_pad = [(0, 0)] * array.ndim
        n_pad[axis] = (0, pad_size)

        return np.pad(array, pad_width=n_pad, mode='constant', constant_values=pad_value)

    def mask_sensors(self, array, axis):

        mask = np.random.randint(low=0, high=2, size=array.shape[axis])
        expanded_mask = np.broadcast_to(mask, array.shape)

        return np.stack(arrays=[array, expanded_mask], axis=-1)

    def create_targets(self, x):
        """ Create target data from potential leak arrays. Outputs both concentrations and categorical (argmax)"""
        q_CH4_concentration = x[:, :, :, -1]
        q_CH4_catergorical = (q_CH4_concentration == q_CH4_concentration.max(axis=1)[:, None]).astype(int)

        return q_CH4_concentration, q_CH4_catergorical










