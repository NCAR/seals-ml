import yaml
import os
import argparse
import glob
from sealsml.data import Preprocessor, save_output
from sealsml.keras.models import (QuantizedTransformer, BlockTransformer, TEncoder,
                                  BackTrackerDNN, LocalizedLeakRateBlockTransformer)
from sealsml.baseline import GPModel
from sealsml.keras.callbacks import LeakLocRateMetricsCallback
from sealsml.backtrack import backtrack_preprocess
from sklearn.model_selection import train_test_split
from os.path import join
import keras
import numpy as np
import datetime
import time
import xarray as xr
import tensorflow as tf
import pandas as pd
from bridgescaler import DQuantileScaler
from sealsml.backtrack import create_binary_preds_relative
from sealsml.keras.metrics import mean_searched_locations
tf.debugging.disable_traceback_filtering()



custom_keras_metrics = {"mean_searched_locations": mean_searched_locations}

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Path to config file")
args = parser.parse_args()
with open(args.config) as config_file:
    config = yaml.safe_load(config_file)
for model in config["models"]:
    if "compile" in config[model]:
        if "metrics" in config[model]["compile"]:
            for m, metric in enumerate(config[model]["compile"]["metrics"]):
                if metric in custom_keras_metrics.keys():
                    config[model]["compile"]["metrics"][m] = custom_keras_metrics[metric]
keras.utils.set_random_seed(config["random_seed"])
np.random.seed(config["random_seed"])
username = os.environ.get('USER')
config["out_path"] = config["out_path"].replace("username", username)
config["data_path"] = config["data_path"].replace("username", username)
date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
out_path = os.path.join(config["out_path"], date_str)
os.makedirs(out_path, exist_ok=False)
with open(join(out_path, 'train.yml'), 'w') as outfile:
     yaml.dump(config, outfile, default_flow_style=False)
files = sorted(glob.glob(os.path.join(config["data_path"], "*.nc")))

training, validation = train_test_split(files,
                                        test_size=config["validation_ratio"],
                                        random_state=config["random_seed"])

p = Preprocessor(scaler_type=config["scaler_type"], sensor_pad_value=-1, sensor_type_value=-999)
p.save_filenames(training, validation, out_path)
start = time.time()
encoder_data, decoder_data, leak_location, leak_rate = p.load_data(training,
                                                                   remove_blind_samples=config["remove_blind_samples"])
print(f"Minutes to load training data: {(time.time() - start) / 60 }")
start = time.time()
fit_scaler = True
if "scaler_path" in config.keys():
    if os.path.exists(join(config["scaler_path"], "coord_scaler.json")):
        p.load_scalers(coord_scaler_path=join(config["scaler_path"], "coord_scaler.json"),
                       sensor_scaler_path=join(config["scaler_path"], "sensor_scaler.json"))
        fit_scaler = False
scaled_encoder, scaled_decoder, encoder_mask, decoder_mask = p.preprocess(encoder_data, decoder_data,
                                                                          fit_scaler=fit_scaler)
print(f"Minutes to fit scaler: {(time.time() - start) / 60 }")
start = time.time()
print(f"Minutes to transform with scaler: {(time.time() - start) / 60 }")
encoder_data_val, decoder_data_val, leak_location_val, leak_rate_val = p.load_data(validation,
                                                                                   remove_blind_samples=config["remove_blind_samples"])
scaled_encoder_val, scaled_decoder_val, encoder_mask_val, decoder_mask_val = p.preprocess(encoder_data_val,
                                                                                          decoder_data_val,
                                                                                          fit_scaler=False)
v = None
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    for model_name in config["models"]:
        start = time.time()
        if model_name == "transformer_leak_loc":
            model = QuantizedTransformer(**config[model_name]["kwargs"])
            y, y_val = leak_location, leak_location_val
        elif model_name == "block_transformer_leak_loc":
            model = BlockTransformer(**config[model_name]["kwargs"])
            y, y_val = leak_location, leak_location_val
        elif model_name == "loc_rate_block_transformer":
            model = LocalizedLeakRateBlockTransformer(**config[model_name]["kwargs"])
            y = (leak_location, leak_rate)
            y_val = (leak_location_val, leak_rate_val)
            cb_metrics = LeakLocRateMetricsCallback((scaled_encoder_val,
                                                    scaled_decoder_val,
                                                    encoder_mask_val,
                                                    decoder_mask_val),
                                                    y_val)
            if "callbacks" not in config[model_name]["fit"].keys():
                config[model_name]["fit"]["callbacks"] = [cb_metrics]
            else:
                config[model_name]["fit"]["callbacks"].append(cb_metrics)
        elif model_name == 'transformer_leak_rate':
            model = TEncoder(**config[model_name]["kwargs"])
            y, y_val = leak_rate, leak_rate_val
        elif model_name == "gaussian_process":
            model = GPModel(**config[model_name]["kwargs"])
            y, y_val = leak_location, leak_location_val
        elif model_name == "backtracker":
            t = xr.open_mfdataset(training, concat_dim='sample', combine="nested", parallel=False)
            v = xr.open_mfdataset(validation, concat_dim='sample', combine="nested", parallel=False)
            model = BackTrackerDNN(**config[model_name]["kwargs"])
            x, y = backtrack_preprocess(t, **config[model_name]["preprocess"])
            x_val, y_val = backtrack_preprocess(v, **config[model_name]["preprocess"])
            scaler = DQuantileScaler()
            scaled_encoder = scaler.fit_transform(x)
            scaled_encoder_val = scaler.transform(x_val)
        else:
            raise ValueError(f"Incompatible model type {model_name}")

        if config[model_name]["optimizer"]["optimizer_type"].lower() == "sgd":
            optimizer = keras.optimizers.SGD(learning_rate=config[model_name]["optimizer"]["learning_rate"],
                            momentum=config[model_name]["optimizer"]["sgd_momentum"])
        elif config[model_name]["optimizer"]["optimizer_type"].lower() == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=config[model_name]["optimizer"]["learning_rate"],
                            beta_1=config[model_name]["optimizer"]["adam_beta_1"],
                            beta_2=config[model_name]["optimizer"]["adam_beta_2"],
                            epsilon=config[model_name]["optimizer"]["epsilon"])
        else:
            optimizer = None
            raise TypeError("Only 'sgd' or 'adam' optimizers are currently supported.")

        model.compile(optimizer=optimizer, **config[model_name]["compile"])
        fit_hist = model.fit(x=(scaled_encoder, scaled_decoder, encoder_mask, decoder_mask),
                            y=y,
                            validation_data=((scaled_encoder_val,
                                            scaled_decoder_val,
                                            encoder_mask_val, decoder_mask_val),
                                            y_val),
                            **config[model_name]["fit"])
        print(f"Minutes to train {model_name} model: {(time.time() - start) / 60 }")
        output = model.predict(x=(scaled_encoder_val, scaled_decoder_val, encoder_mask_val, decoder_mask_val),
                            batch_size=config["predict_batch_size"])
        output_train = model.predict(x=(scaled_encoder, scaled_decoder, encoder_mask, decoder_mask),
                                    batch_size=config["predict_batch_size"])

        if model_name == "backtracker":
            backtracker_targets = create_binary_preds_relative(v, output)
            pd.DataFrame(y, columns=['x', 'y', 'z', 'leakrate']).to_csv(os.path.join(out_path, 'seals_train_true.csv'),
                                                                        index=False)
            pd.DataFrame(output_train, columns=['x', 'y', 'z', 'leakrate']).to_csv(
                        os.path.join(out_path, 'seals_train_preds.csv'),
                        index=False)
            pd.DataFrame(y_val, columns=['x', 'y', 'z', 'leakrate']).to_csv(os.path.join(out_path, 'seals_val_true.csv'),
                                                                            index=False)
            pd.DataFrame(output, columns=['x', 'y', 'z', 'leakrate']).to_csv(os.path.join(out_path, 'seals_val_preds.csv'),
                                                                            index=False)

        scalers_saved = False
        if config["save_model"]:
            if model_name != "gaussian_process":
                model.save(os.path.join(out_path, f"{model_name}_{date_str}.keras"))
            if not scalers_saved:
                p.save_scalers(out_path)
                scalers_saved = True

        if config["save_output"]:
            if model_name != "backtracker":
                save_output(out_path=os.path.join(out_path, f"{model_name}_output_{date_str}.nc"),
                            train_targets=y,
                            val_targets=y_val,
                            train_predictions=output_train,
                            val_predictions=output,
                            model_name=model_name)
            loss_hist = pd.DataFrame(fit_hist.history)
            loss_hist.to_csv(os.path.join(out_path, f"{model_name}_model_hist_{date_str}.csv"))
        print('completed.')
# save seperate scaler for back_tracker?
# save output for backtracker as NETCDF as same format as others (added x, y, z)
# Add number of sensors as config option for back tracker
#
