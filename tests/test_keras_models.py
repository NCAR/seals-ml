from sealsml.keras.layers import VectorQuantizer
from sealsml.keras.models import QuantizedTransformer
import numpy as np


def test_quantized_transformer():
    from keras.models import load_model
    batch_size = 256
    train_size = 1024
    encoder_sensor_size = 12
    encoder_seq_size = 100
    decoder_seq_size = 5
    n_vars = 5
    np.random.seed(32525)
    x_encoder = np.random.random((train_size, encoder_sensor_size, encoder_seq_size, n_vars))
    x_decoder = np.random.random((train_size, decoder_seq_size, n_vars))
    y = np.zeros((train_size, decoder_seq_size))
    y[x_decoder[:, :, 1].argmax(axis=0)] = 1
    qt = QuantizedTransformer(use_quantizer=True, encoder_layers=2, decoder_layers=2)
    qt.compile(loss="binary_crossentropy", optimizer="adam")
    qt.call((x_encoder, x_decoder))
    qt.fit((x_encoder, x_decoder), y, batch_size=batch_size, epochs=1)
    weights_init = qt.get_weights()
    qt.fit((x_encoder, x_decoder), y, batch_size=batch_size, epochs=1)
    weights_after = qt.get_weights()
    weights_constant = [np.all(s == e) for s, e in zip(weights_init, weights_after)]
    assert not np.any(weights_constant), "Weights are not changing somewhere"
    y_pred = qt.predict([x_encoder, x_decoder], batch_size=train_size)
    assert y_pred[:, :, 0].shape == y.shape
    qt.save("./test_model.keras")
    new_qt = load_model("./test_model.keras", custom_objects={"VectorQuantizer": VectorQuantizer})
    y_pred_new = new_qt.predict([x_encoder, x_decoder], batch_size=train_size)
    max_pred_diff = np.max(np.abs(y_pred - y_pred_new))
    assert max_pred_diff == 0, f"predictions change by max {max_pred_diff}"
    print()
    print(qt.summary())
    print(new_qt.summary())
    assert new_qt.summary() == qt.summary(), "models do not match"

