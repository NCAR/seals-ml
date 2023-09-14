import xarray as xr
import numpy as np

class DataSampler(object):

    def __init__(self, min_trace_sensors=3, max_trace_sensors=15, n_leaks=1, reference_loc=(15, 15),
                 sensor_mask_radius=50, sensor_height=3, leak_radius=10, emission_var='q_CH4'):

        self.min_trace_sensors = min_trace_sensors
        self.max_trace_sensors = max_trace_sensors
        self.n_leaks = n_leaks
        self.reference_loc = reference_loc
        self.sensor_mask_radius = sensor_mask_radius
        self.sensor_height = sensor_height
        self.leak_radius = leak_radius
        self.emission_var = emission_var

    def load_data(self, file_names):

        self.data = xr.open_mfdataset(file_names, parallel=True)
        self.data = self.data.assign_coords({'iDim': self.data['iDim'],
                                             'jDim': self.data['jDim'],
                                             'kDim': self.data['kDim']})

        self.time_steps = len(self.data['timeDim'].values)

    def sample(self, window_size, samples_per_window):

        for t in np.arange(0, self.time_steps - window_size):

            n_sensors = np.random.randint(low=self.min_trace_sensors,
                                          high=self.max_trace_sensors,
                                          size=samples_per_window)

            i = np.random.randint(low=0, high=30, size=n_sensors)
            j = np.random.randint(low=0, high=30, size=n_sensors)

            sensor_sample = self.data[['u', 'v', 'w', self.emission_var]].isel(timeDim=slice(t, t + window_size),
                                                                               kDim=self.sensor_height,
                                                                               jDim=j,
                                                                               iDim=i).to_array()
