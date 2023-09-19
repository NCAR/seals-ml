import xarray as xr
import numpy as np

class DataSampler(object):

    def __init__(self, min_trace_sensors=3, max_trace_sensors=15, n_leaks=1, reference_loc=(15, 15),
                 sensor_mask_radius=50, sensor_height=3, leak_radius=10, coord_vars=('xPos', 'yPos', 'zPos'),
                 met_vars=('u', 'v', 'w'), emission_vars='q_CH4'):

        self.min_trace_sensors = min_trace_sensors
        self.max_trace_sensors = max_trace_sensors
        self.n_leaks = n_leaks
        self.reference_loc = reference_loc
        self.sensor_mask_radius = sensor_mask_radius
        self.sensor_height = sensor_height
        self.leak_radius = leak_radius
        self.coord_vars = coord_vars
        self.met_vars = met_vars
        self.emission_vars = emission_vars

    def load_data(self, file_names):

        self.data = xr.open_mfdataset(file_names, parallel=True).load()
        self.time_steps = len(self.data['timeDim'].values)

    def sample(self, time_window_size, samples_per_window):

        arrays = []
        
        for t in np.arange(0, self.time_steps - time_window_size):
            
            for _ in range(samples_per_window):
                
                n_sensors = np.random.randint(low=self.min_trace_sensors, high=self.max_trace_sensors)

                i = np.random.randint(low=0, high=30, size=n_sensors)
                j = np.random.randint(low=0, high=30, size=n_sensors)
                sensor_sample = self.data[['xPos', 'yPos', 'zPos', 'u']].to_array().expand_dims('sample').values[:, :, 0, i, j, t:t + time_window_size]

                padded_sample = self.pad_along_axis(sensor_sample, target_length=self.max_trace_sensors, pad_value=0, axis=2)
                arrays.append(padded_sample)
            
        return np.vstack(arrays)
        
        
    def pad_along_axis(self, array: np.ndarray, target_length: int, pad_value: int = 0, axis: int = 0) -> np.ndarray:

        pad_size = target_length - array.shape[axis]

        if pad_size <= 0:
            return array

        npad = [(0, 0)] * array.ndim
        npad[axis] = (0, pad_size)

        return np.pad(array, pad_width=npad, mode='constant', constant_values=pad_value)
