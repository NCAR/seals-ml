from sealsml.keras.models import QuantizedTransformer, TEncoder, BackTrackerDNN
from sealsml.data import Preprocessor
from sealsml.backtrack import preprocess, create_binary_preds_relative
import numpy as np
import xarray as xr
from keras.models import load_model

test_data = ["../test_data/test_data_CBL2m_Ug10_src1-8kg_a.3.nc"]
p = Preprocessor(scaler_type="quantile", sensor_pad_value=-1, sensor_type_value=-999)
encoder_data, decoder_data, y, y_leak_rate = p.load_data(test_data)
x_encoder, encoder_mask = p.preprocess(encoder_data, fit_scaler=True)
x_decoder, decoder_mask = p.preprocess(decoder_data, fit_scaler=False)
batch_size = x_encoder.shape[0]

def test_quantized_transformer():

    np.random.seed(32525)
    print("x encoder shape", x_encoder.shape)
    print("x decoder shape", x_decoder.shape)
    qt = QuantizedTransformer(use_quantizer=True, encoder_layers=2, decoder_layers=2)
    qt.compile(loss="binary_crossentropy", optimizer="adam")
    qt.call((x_encoder, x_decoder))
    weights_init = qt.get_weights()
    qt.fit((x_encoder, x_decoder), y, batch_size=batch_size, epochs=1)
    weights_after = qt.get_weights()
    weights_constant = [np.all(s == e) for s, e in zip(weights_init, weights_after)]
    assert not np.any(weights_constant), "Weights are not changing somewhere"
    y_pred = qt.predict([x_encoder, x_decoder], batch_size=batch_size)
    assert y_pred[:, :, 0].shape == y.shape
    qt.save("./test_model.keras")
    new_qt = load_model("./test_model.keras", custom_objects={"VectorQuantizer": VectorQuantizer})
    y_pred_new = new_qt.predict([x_encoder, x_decoder], batch_size=batch_size)
    max_pred_diff = np.max(np.abs(y_pred - y_pred_new))
    assert max_pred_diff == 0, f"predictions change by max {max_pred_diff}"
    assert new_qt.summary() == qt.summary(), "models do not match"

def test_transformer_regressor():
    np.random.seed(32525)
    print("x encoder shape", x_encoder.shape)
    print("x decoder shape", x_decoder.shape)
    model = TEncoder(use_quantizer=True, encoder_layers=2)
    model.compile(loss="mse", optimizer="adam")
    model.call((x_encoder, x_decoder))
    weights_init = model.get_weights()
    model.fit((x_encoder, x_decoder), y_leak_rate, batch_size=batch_size, epochs=1)
    weights_after = model.get_weights()
    weights_constant = [np.all(s == e) for s, e in zip(weights_init, weights_after)]
    assert not np.any(weights_constant), "Weights are not changing somewhere"
    y_pred = model.predict([x_encoder, x_decoder], batch_size=batch_size)
    assert y_pred[:, 0].shape == y_leak_rate.shape
    model.save("./test_model_2.keras")
    new_model = load_model("./test_model_2.keras", custom_objects={"VectorQuantizer": VectorQuantizer})
    y_pred_new = new_model.predict([x_encoder, x_decoder], batch_size=batch_size)
    max_pred_diff = np.max(np.abs(y_pred - y_pred_new))
    assert max_pred_diff == 0, f"predictions change by max {max_pred_diff}"
    assert new_model.summary() == model.summary(), "models do not match"


def test_backtracker():
    data = xr.open_dataset(test_data[0])
    x, y = preprocess(xr.open_dataset(test_data[0]), n_sensors=3)
    np.random.seed(32525)
    model = BackTrackerDNN(hidden_layers=2, hidden_neurons=128)
    weights_init = model.get_weights()
    model.fit(x=(x, None, None, None), y=y, epochs=100)
    weights_after = model.get_weights()
    weights_constant = [np.all(s == e) for s, e in zip(weights_init, weights_after)]
    assert not np.any(weights_constant), "Weights are not changing somewhere"
    y_pred = model.predict([x, x_decoder], batch_size=batch_size)
    y_pred_target = create_binary_preds_relative(data, y_pred)
    assert y_pred_target.shape == data['target'].squeeze().shape
    model.save("./test_model_3.keras")
    new_model = load_model("./test_model_3.keras")
    y_pred_new = new_model.predict([x, x_decoder], batch_size=batch_size)
    max_pred_diff = np.max(np.abs(y_pred - y_pred_new))
    assert max_pred_diff == 0, f"predictions change by max {max_pred_diff}"
    assert new_model.summary() == model.summary(), "models do not match"
