import yaml
import os
import argparse
import glob
from sealsml.data import Preprocessor, save_output
from bridgescaler import save_scaler
from sealsml.keras.models import QuantizedTransformer, TEncoder, BackTrackerDNN
from sealsml.baseline import GPModel
from sealsml.backtrack import preprocess
from sklearn.model_selection import train_test_split
import keras
import numpy as np
import datetime
import time
import xarray as xr
import tensorflow as tf
import pandas as pd
from bridgescaler import DeepQuantileTransformer, DeepMinMaxScaler, DeepStandardScaler
from sealsml.backtrack import create_binary_preds_relative
from sealsml.evaluate import provide_metrics
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

print('files=',files)

training, validation = train_test_split(files,
                                        test_size=config["validation_ratio"],
                                        random_state=config["random_seed"])

p = Preprocessor(scaler_type=config["scaler_type"], sensor_pad_value=-1, sensor_type_value=-999)
p.save_filenames(training, validation, config["out_path"])
start = time.time()
encoder_data, decoder_data, leak_location, leak_rate = p.load_data(training)
print(f"Minutes to load training data: {(time.time() - start) / 60 }")
start = time.time()
scaled_encoder, encoder_mask = p.preprocess(encoder_data, fit_scaler=True)
print(f"Minutes to fit scaler: {(time.time() - start) / 60 }")
start = time.time()
scaled_decoder, decoder_mask = p.preprocess(decoder_data, fit_scaler=False)
print(f"Minutes to transform with scaler: {(time.time() - start) / 60 }")
encoder_data_val, decoder_data_val, leak_location_val, leak_rate_val = p.load_data(validation)
scaled_encoder_val, encoder_mask_val = p.preprocess(encoder_data_val, fit_scaler=False)
scaled_decoder_val, decoder_mask_val = p.preprocess(decoder_data_val, fit_scaler=False)
date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
out_path = os.path.join(config["out_path"], date_str)
os.makedirs(out_path, exist_ok=False)

for model_name in config["models"]:
    start = time.time()
    if model_name == "transformer_leak_loc":
        model = QuantizedTransformer(**config[model_name]["kwargs"])
        model.compile(**config[model_name]["compile"])
        y, y_val = leak_location, leak_location_val
    elif model_name == 'transformer_leak_rate':
        model = TEncoder(**config[model_name]["kwargs"])
        model.compile(**config[model_name]["compile"])
        y, y_val = leak_rate, leak_rate_val
    elif model_name == "gaussian_process":
        model = GPModel(**config[model_name]["kwargs"])
        y, y_val = leak_location, leak_location_val
    elif model_name == "backtracker":
        model = BackTrackerDNN(**config[model_name]["kwargs"])

        model.compile(**config[model_name]["compile"])

        x, y = preprocess(xr.open_mfdataset(training, concat_dim='sample', combine="nested", parallel=False),
                          n_sensors=4)
        x_val, y_val = preprocess(xr.open_mfdataset(validation, concat_dim='sample', combine="nested", parallel=False),
                                  n_sensors=4)

        print('x,y,xval,yval.shapes=',x.shape,y.shape,x_val.shape,y_val.shape)

        scaler = DeepQuantileTransformer()
        scaled_encoder = scaler.fit_transform(x)
        scaled_encoder_val = scaler.transform(x_val)
    else:
        raise ValueError(f"Incompatible model type {model_name}")
    fit_hist = model.fit(x=(scaled_encoder, scaled_decoder, encoder_mask, decoder_mask),
                         y=y,
                         validation_data=((scaled_encoder_val,
                                           scaled_decoder_val,
                                           encoder_mask_val,
                                           decoder_mask_val),
                                          y_val),
                         **config[model_name]["fit"])
    print(f"Minutes to train {model_name} model: {(time.time() - start) / 60 }")
    output = model.predict(x=(scaled_encoder_val, scaled_decoder_val, encoder_mask_val, decoder_mask_val),
                           batch_size=config["predict_batch_size"]).squeeze()
    output_train = model.predict(x=(scaled_encoder, scaled_decoder, encoder_mask, decoder_mask),
                                 batch_size=config["predict_batch_size"]).squeeze()

    if model_name == "backtracker":
        backtracker_targets = create_binary_preds_relative(validation, output)
        pd.DataFrame(y, columns=['x', 'y', 'z', 'leakrate']).to_csv(os.path.join(out_path, 'seals_train_true.csv'))
        pd.DataFrame(output_train, columns=['x', 'y', 'z', 'leakrate']).to_csv(
            os.path.join(out_path, 'seals_train_preds.csv'))
        pd.DataFrame(y_val, columns=['x', 'y', 'z', 'leakrate']).to_csv(os.path.join(out_path, 'seals_val_true.csv'))
        pd.DataFrame(output, columns=['x', 'y', 'z', 'leakrate']).to_csv(os.path.join(out_path, 'seals_val_preds.csv'))

    scaler_saved = False
    if config["save_model"]:
        if model_name != "gaussian_process":
            model.save(os.path.join(out_path, f"{model_name}_{date_str}.keras"))
        if not scaler_saved:
            save_scaler(p.scaler, os.path.join(out_path, f"scaler_{date_str}.json"))
            scaler_saved = True

    if config["save_output"]:

        save_output(out_path=os.path.join(out_path, f"{model_name}_output_{date_str}.nc"),
                    train_targets=y,
                    val_targets=y_val,
                    train_predictions=output_train,
                    val_predictions=output,
                    model_name=model_name)
        loss_hist = pd.DataFrame(fit_hist.history)
        loss_hist.to_csv(os.path.join(out_path, f"{model_name}_model_hist_{date_str}.csv"))

