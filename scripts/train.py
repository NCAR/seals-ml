import yaml
import os
import argparse
import glob
from sealsml.data import Preprocessor
from bridgescaler import save_scaler
from sealsml.keras.models import QuantizedTransformer
from sealsml.baseline import GPModel
from sealsml.evaluate import provide_metrics
from sklearn.model_selection import train_test_split
import keras
import numpy as np
import xarray as xr
import datetime
import time
import tensorflow as tf
import pandas as pd
tf.debugging.disable_traceback_filtering()

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

training, validation = train_test_split(files[:3],
                                        test_size=config["validation_ratio"],
                                        random_state=config["random_seed"])

p = Preprocessor(scaler_type=config["scaler_type"], sensor_pad_value=-1, sensor_type_value=-999)
p.save_filenames(training, validation, config["out_path"])
start = time.time()
encoder_data, decoder_data, targets = p.load_data(training)
print(f"Minutes to load training data: {(time.time() - start) / 60 }")
start = time.time()
scaled_encoder, encoder_mask = p.preprocess(encoder_data, fit_scaler=True)
print(f"Minutes to fit scaler: {(time.time() - start) / 60 }")
start = time.time()
scaled_decoder, decoder_mask = p.preprocess(decoder_data, fit_scaler=False)
print(f"Minutes to transform with scaler: {(time.time() - start) / 60 }")
encoder_data_val, decoder_data_val, targets_val = p.load_data(validation)
scaled_encoder_val, encoder_mask_val = p.preprocess(encoder_data_val, fit_scaler=False)
scaled_decoder_val, decoder_mask_val = p.preprocess(decoder_data_val, fit_scaler=False)
print("encoder mask:", encoder_mask.shape)
print("decoder mask:", decoder_mask.shape)
print("encoder:", scaled_encoder.shape)
print("decoder:", scaled_decoder[..., :4].shape)
print(targets.shape)

date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
out_path = os.path.join(config["out_path"], date_str)
os.makedirs(out_path, exist_ok=False)

for model_name in config["models"]:
    start = time.time()

    if model_name == "transformer":
        model = QuantizedTransformer(**config[model_name]["kwargs"])
        model.compile(**config[model_name]["compile"])
    elif model_name == "gaussian_process":
        model = GPModel(**config[model_name]["kwargs"])
    elif model_name == "back_tracker":
        continue
    fit_hist = model.fit(x=(scaled_encoder, scaled_decoder, encoder_mask, decoder_mask),
                         y=targets,
                         validation_data=((scaled_encoder_val,
                                           scaled_decoder_val,
                                           encoder_mask_val,
                                           decoder_mask_val),
                                          targets_val),
                         **config[model_name]["fit"])
    print(f"Minutes to train {model_name} model: {(time.time() - start) / 60 }")
    output = model.predict(x=(scaled_encoder_val, scaled_decoder_val, encoder_mask_val, decoder_mask_val),
                           batch_size=config["predict_batch_size"]).squeeze()
    metrics = provide_metrics(targets_val, output)
    print(metrics)
    scaler_saved = False
    if config["save_model"]:
        if model_name != "gaussian_process":
            model.save(os.path.join(out_path, f"{model_name}_{date_str}.keras"))
        if not scaler_saved:
            save_scaler(p.scaler, os.path.join(out_path, f"scaler_{date_str}.json"))
            scaler_saved = True

    if config["save_output"]:

        output = xr.Dataset(data_vars=dict(targets=(["sample", "pot_leak_locs"], targets_val),
                                           probabilities=(["sample", "pot_leak_locs"], output)))
        output.to_netcdf(os.path.join(out_path, f"model_output_{date_str}.nc"))
        # loss_hist = pd.DataFrame(fit_hist.history)
        # loss_hist.to_csv(os.path.join(config["out_path"], f"model_hist_{date_str}.csv"))

