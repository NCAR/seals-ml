from sealsml.keras.layers import TimeBlockSensorEncoder

import numpy as np

def test_time_block_sensor_encoder():
    x_shape = (128, 5, 100, 3)
    x_test = np.random.random(x_shape)
    y = x_test.reshape(x_shape[0], x_shape[1] * x_shape[2] * x_shape[3]).mean(axis=1)
    tbse = TimeBlockSensorEncoder(embedding_size=128, block_size=10)
    tbse_out = tbse(x_test)
    return