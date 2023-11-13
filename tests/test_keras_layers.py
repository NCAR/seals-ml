from sealsml.keras.layers import ConvSensorEncoder
from keras.models import Sequential
from keras.layers import Flatten, Dense
import numpy as np

def test_convsensorencoder():
    min_filters = 8
    x_shape = (128, 5, 100, 3)
    x_test = np.random.random(x_shape)
    y = x_test.reshape(x_shape[0], x_shape[1] * x_shape[2] * x_shape[3]).mean(axis=1)
    conv_encoder = ConvSensorEncoder(min_filters=min_filters)
    conv_config = conv_encoder.get_config()
    x_encoded = conv_encoder(x_test)
    assert len(x_encoded.shape) == 3
    assert x_encoded.shape[0] == x_shape[0] and x_encoded.shape[1] == x_shape[1]
    assert conv_config["min_filters"] == min_filters
    model = Sequential(layers=[ConvSensorEncoder(min_filters=min_filters, input_shape=x_shape[1:]),
                               Flatten(), Dense(1)])
    model.compile(loss="mse", optimizer="adam")
    model.fit(x_test, y, epochs=10)
    y_pred = model.predict(x_test)
    assert y_pred.shape == (x_shape[0], 1)
    return