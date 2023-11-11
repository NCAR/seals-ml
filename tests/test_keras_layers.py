from sealsml.keras.layers import ConvSensorEncoder
import numpy as np

def test_convsensorencoder():
    min_filters = 8
    x_shape = (8, 5, 100, 3)
    x_test = np.random.random(x_shape)
    conv_encoder = ConvSensorEncoder(min_filters=min_filters)
    conv_config = conv_encoder.get_config()
    x_encoded = conv_encoder(x_test)
    assert len(x_encoded.shape) == 3
    assert x_encoded.shape[0] == x_shape[0] and x_encoded.shape[1] == x_shape[1]
    assert conv_config["min_filters"] == min_filters
    return