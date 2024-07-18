from echo.src.base_objective import BaseObjective
import numpy as np
from .models import BlockTransformer, LocalizedLeakRateBlockTransformer, QuantizedTransformer, TEncoder, BackTrackerDNN
import keras
import os
import datetime
import time
from os.path import join
from .metrics import mean_searched_locations
from sealsml.data import Preprocessor, save_output
from sklearn.model_selection import train_test_split
import glob
from bridgescaler import DQuantileScaler
import xarray as xr
import pandas as pd
from .callbacks import LeakLocRateMetricsCallback
from sealsml.backtrack import backtrack_preprocess, create_binary_preds_relative

class Objective(BaseObjective):
    def __init__(self, config, metric="val_loss"):
        BaseObjective.__init__(self, config, metric)

    def train(self, trial, config):
        custom_keras_metrics = {"mean_searched_locations": mean_searched_locations}
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
        date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
        files = glob.glob(os.path.join(config["data_path"], "*.nc"))

        training, validation = train_test_split(files,
                                                test_size=config["validation_ratio"],
                                                random_state=config["random_seed"])

        p = Preprocessor(scaler_type=config["scaler_type"], sensor_pad_value=-1, sensor_type_value=-999)
        start = time.time()
        encoder_data, decoder_data, leak_location, leak_rate = p.load_data(training)
        print(f"Minutes to load training data: {(time.time() - start) / 60}")
        start = time.time()
        scaled_encoder, scaled_decoder, encoder_mask, decoder_mask = p.preprocess(encoder_data, decoder_data,
                                                                                  fit_scaler=True)
        print(f"Minutes to fit scaler: {(time.time() - start) / 60}")
        start = time.time()
        print(f"Minutes to transform with scaler: {(time.time() - start) / 60}")
        encoder_data_val, decoder_data_val, leak_location_val, leak_rate_val = p.load_data(validation)
        scaled_encoder_val, scaled_decoder_val, encoder_mask_val, decoder_mask_val = p.preprocess(encoder_data_val,
                                                                                                  decoder_data_val,
                                                                                                  fit_scaler=False)
        model_name = config["models"][0]
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
        print(f"Minutes to train {model_name} model: {(time.time() - start) / 60}")
        fit_output = {}
        for k, v in fit_hist.history.items():
            if isinstance(fit_hist.history[k], list):
                fit_output[k] = v[-1]
        return fit_output
