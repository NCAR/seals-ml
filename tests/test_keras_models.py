from sealsml.keras.models import QuantizedTransformer
import numpy as np

def test_quantized_transformer():
    batch_size = 32
    train_size= 1024
    encoder_seq_size = 10
    decoder_seq_size = 5
    n_vars = 30
    x_encoder = np.random.random((train_size, encoder_seq_size, n_vars))
    x_decoder = np.random.random((train_size, decoder_seq_size, n_vars))
    y = np.zeros((train_size, decoder_seq_size))
    y[x_decoder[:, :, 1].argmax(axis=1)] = 1
    qt = QuantizedTransformer()
    qt.compile(loss="binary_crossentropy", optimizer="adam")
    y_pred = qt.predict([x_encoder, x_decoder])
    print(y_pred)
    assert y_pred[:, :, 0].shape == y.shape
    qt.fit([x_encoder, x_decoder], y, batch_size=batch_size)