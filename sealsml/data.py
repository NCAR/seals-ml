import xarray as xr
import numpy as np

class DataSampler(object):
    """ Sample LES data with various geometric configurations. """

    def __init__(self, min_trace_sensors=3, max_trace_sensors=15, min_leak_loc=1, max_leak_loc=10,
                 sensor_mask_radius=50, sensor_height=3, leak_radius=10,
                 coord_vars=["ref_distance", "ref_azi", "ref_elv"],
                 met_vars=['u', 'v', 'w'], emission_vars=['q_CH4']):

        self.min_trace_sensors = min_trace_sensors
        self.max_trace_sensors = max_trace_sensors
        self.min_leak_loc = min_leak_loc
        self.max_leak_loc = max_leak_loc
        self.sensor_mask_radius = sensor_mask_radius
        self.sensor_height = sensor_height
        self.leak_radius = leak_radius
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
        self.resolution = self.data['xPos'][0, 0, 1].values - self.data['xPos'][0, 0, 0].values
        for var in ["ref_distance", "ref_azi", "ref_elv"]:
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
            Numpy Array of shape (sample, variable, sensor, time) """

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
                                k, i_sensor, j_sensor, t:t + time_window_size]
                leak_sample = self.data[self.variables].to_array().expand_dims('sample').values[:, :,
                                k, i_leak, j_leak, t + time_window_size: t + time_window_size + 1]

                sensor_dist, sensor_azi = self.derive_variables(i_sensor, j_sensor, np.repeat(k, n_sensors),
                                                                reference_point)
                leak_dist, leak_azi = self.derive_variables(i_leak, j_leak, np.repeat(k, n_leaks), reference_point)

                sensor_sample[0, 0, :] = np.broadcast_to(sensor_dist, shape=(time_window_size, sensor_dist.shape[0])).T
                sensor_sample[0, 1, :] = np.broadcast_to(sensor_azi, shape=(time_window_size, sensor_azi.shape[0])).T

                leak_sample[0, 0, :, 0] = leak_dist
                leak_sample[0, 1, :, 0] = leak_azi

                padded_sensor_sample = self.pad_along_axis(sensor_sample, target_length=self.max_trace_sensors,
                                                           pad_value=0, axis=2)
                padded_leak_sample = self.pad_along_axis(leak_sample, target_length=self.max_leak_loc,
                                                         pad_value=0, axis=2)

                # target = padded_leak_sample[:, -1, ...] # can be done after stacking in separate method

                sensor_arrays.append(padded_sensor_sample)
                leak_arrays.append(padded_leak_sample)

        sensor_samples = np.transpose(np.vstack(sensor_arrays), axes=[0, 2, 3, 1])
        leak_samples = np.transpose(np.vstack(leak_arrays), axes=[0, 2, 3, 1])

        return sensor_samples, leak_samples

    def derive_variables(self, i, j, k, reference_point):

        sample_points = np.stack([i, j, k]).T
        reference_points = np.broadcast_to(reference_point, shape=sample_points.shape)

        distances = self.calc_euclidean_dist(sample_points, reference_points)
        azimuths = self.calc_azimuth(sample_points, reference_points)

        return distances, azimuths

    def calc_euclidean_dist(self, sample_array, reference_array, axis=1):

        return np.linalg.norm(sample_array - reference_array, axis=axis)

    def calc_azimuth(self, sample_array, reference_array):

        reference, points = reference_array[:, :-1], sample_array[:, :-1]  # remove k dimension
        diff = reference - points
        angle_degree = np.degrees(np.arctan(diff[:, 1].astype('float32'), diff[:, 0].astype('float32')))

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








