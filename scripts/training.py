import numpy as np
import glob
from sealsml.keras.models import QuantizedTransformer
from sealsml.data import Preprocessor
from sealsml.evaluate import provide_metrics

files = glob.glob("/Users/cbecker/PycharmProjects/seals-ml/data/test_sample_data_10172025/*.nc")

p = Preprocessor(scaler_type="quantile", sensor_pad_value=-999, sensor_type_value=-1)
encoder_data, decoder_data, targets = p.load_data(files[:4])
scaled_encoder, encoder_mask = p.preprocess(encoder_data, fit_scaler=True)
scaled_decoder, decoder_mask = p.preprocess(decoder_data, fit_scaler=False)
model = QuantizedTransformer(use_quantizer=True, encoder_layers=2, decoder_layers=2)
model.compile(loss="binary_crossentropy", optimizer="adam")

model.fit((scaled_encoder, scaled_decoder, encoder_mask, decoder_mask),
          targets, verbose=1, epochs=5, batch_size=1024)

probs = model.predict((scaled_encoder, scaled_decoder, encoder_mask, decoder_mask), batch_size=10000).squeeze()

metrics = provide_metrics(targets.squeeze(), probs)

print(metrics)
