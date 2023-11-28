import yaml
import os
import argparse
import glob
from sealsml.keras.models import QuantizedTransformer
from sealsml.data import Preprocessor
from sealsml.evaluate import provide_metrics
from sklearn.model_selection import train_test_split
from bridgescaler import save_scaler
import keras
import numpy as np
import xarray as xr
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Path to config file")
args = parser.parse_args()
with open(args.config) as config_file:
    config = yaml.safe_load(config_file)

keras.utils.set_random_seed(config["random_seed"])
np.random.seed(config["random_seed"])
username = os.environ.get('USER')
config["out_path"] = config["out_path"].replace("username", username)

files = glob.glob(os.path.join(config["data_path"], "*.nc"))
print(files)
training, validation = train_test_split(files,
                                        test_size=config["validation_ratio"],
                                        random_state=config["random_seed"])

p = Preprocessor(scaler_type=config["scaler_type"], sensor_pad_value=-1, sensor_type_value=-999)

encoder_data, decoder_data, targets = p.load_data(training)
scaled_encoder, encoder_mask = p.preprocess(encoder_data, fit_scaler=True)
scaled_decoder, decoder_mask = p.preprocess(decoder_data, fit_scaler=False)

encoder_data_val, decoder_data_val, targets_val = p.load_data(validation)
scaled_encoder_val, encoder_mask_val = p.preprocess(encoder_data_val, fit_scaler=False)
scaled_decoder_val, decoder_mask_val = p.preprocess(decoder_data_val, fit_scaler=False)

model = QuantizedTransformer(**config["model"])
model.compile(**config["model_compile"])

model.fit(x=(scaled_encoder, scaled_decoder, encoder_mask, decoder_mask),
          y=targets,
          validation_data=((scaled_encoder_val, scaled_decoder_val, encoder_mask_val, decoder_mask_val), targets_val),
          **config["model_fit"])

probabilities = model.predict(x=(scaled_encoder_val, scaled_decoder_val, encoder_mask_val, decoder_mask_val),
                              batch_size=config["predict_batch_size"]).squeeze()

metrics = provide_metrics(targets_val, probabilities)

print(metrics)
date_str = datetime.datetime.now().strftime("%Y-%m-%d")

if config["save_model"]:

    os.makedirs(config["out_path"], exist_ok=True)
    save_scaler(p.scaler, os.path.join(config["out_path"], f"scaler_{date_str}.json"))
    model.save(os.path.join(config["out_path"], f"model_{date_str}.keras"))

if config["save_output"]:

    output = xr.Dataset(data_vars=dict(targets=(["sample", "pot_leak_locs"], targets_val),
                                       probabilities=(["sample", "pot_leak_locs"], probabilities)))
    output.to_netcdf(os.path.join(config["out_path"], f"model_output_{date_str}.nc"))

