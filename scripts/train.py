import yaml
import os
import argparse
import glob
from sealsml.data import Preprocessor, save_output
from sealsml.keras.models import (BlockTransformer, BlockEncoder,
                                  BackTrackerDNN, LocalizedLeakRateBlockTransformer)
from sealsml.baseline import GPModel
from sealsml.keras.callbacks import LeakLocRateMetricsCallback, LeakLocMetricsCallback, LeakRateMetricsCallback
from sealsml.backtrack import backtrack_preprocess, backtrack_scaleDataTuple, backtrack_unscaleDataTuple, mapPredlocsToClosestPL
from sealsml.backtrack import scalings_bjt, scaler_bjt_x, scaler_bjt_y, scaler_bjt_y_inverse, truth_values
from sklearn.model_selection import train_test_split
from os.path import join
import keras
import numpy as np
import datetime
import time
import xarray as xr
import tensorflow as tf
import pandas as pd
from bridgescaler import save_scaler
from sealsml.backtrack import create_binary_preds_relative
import keras.models as models
from keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint
tf.debugging.disable_traceback_filtering()

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Path to config file")
args = parser.parse_args()
with open(args.config) as config_file:
    config = yaml.safe_load(config_file)

keras.utils.set_random_seed(config["random_seed"])
np.random.seed(config["random_seed"])
username = os.environ.get('USER')
date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
if config["restart_options"]["restart"]:
    out_path = config["restart_options"]["model_out_path"]
elif not config["restart_options"]["restart"]:
    config["out_path"] = config["out_path"].replace("username", username)
    config["data_path"] = config["data_path"].replace("username", username)
    out_path = os.path.join(config["out_path"], date_str)
    os.makedirs(out_path, exist_ok=False)

print("Path to save all results:", out_path)
with open(join(out_path, 'train.yml'), 'w') as outfile:
    yaml.dump(config, outfile, default_flow_style=False)

files = sorted(glob.glob(os.path.join(config["data_path"], "*.nc")))

training, validation = train_test_split(files,
                                        test_size=config["validation_ratio"],
                                        random_state=config["random_seed"])
#Construct a preprocessor class instance
p = Preprocessor(scaler_type=config["scaler_type"],
                 sensor_pad_value=-1,
                 sensor_type_value=-999,
                 scaler_options=config["scaler_options"])
p.save_filenames(training, validation, out_path)

#Load predetermined scaler fit configurations if desired
fit_scaler = True
scalers_saved = False
if "scaler_path" in config.keys():
    if os.path.exists(join(config["scaler_path"], "coord_scaler.json")):
        p.load_scalers(coord_scaler_path=join(config["scaler_path"], "coord_scaler.json"),
                       sensor_scaler_path=join(config["scaler_path"], "sensor_scaler.json"))
        fit_scaler = False

#Preprocess (load and scale data) in transformer or backtracker mode 
preproc_done = False
for model_name in config["models"]:
    if "backtracker" in model_name:
      start = time.time()
      print(f"contains backtracker {model_name}")
      t = xr.open_mfdataset(training, concat_dim='sample', combine="nested", parallel=False)
      print(f"Minutes to load training data: {(time.time() - start) / 60}")
      start = time.time()
      x, speed, L_scale, H_scale, n_samples, n_pot_leaks, x_pot_leaks, y_pot_leaks, z_pot_leaks = backtrack_preprocess(t, **config[model_name]["preprocess"])
      print(f"Minutes to run backtrack_preprocess on training data: {(time.time() - start) / 60}")
      start = time.time()
      y = truth_values(t)
      print(f"Minutes to preprocess truth_values of training data: {(time.time() - start) / 60}")
      start = time.time()

      v = xr.open_mfdataset(validation, concat_dim='sample', combine="nested", parallel=False)
      print(f"Minutes to load validation data: {(time.time() - start) / 60}")
      start = time.time()
      x_val, speed_val, L_scale_val, H_scale_val, n_samples_val, n_pot_leaks_val, x_pot_leaks_val, y_pot_leaks_val, z_pot_leaks_val = backtrack_preprocess(v, **config[model_name]["preprocess"])
      print(f"Minutes to run backtrack_preprocess on validation data: {(time.time() - start) / 60}")
      start = time.time()
      y_val = truth_values(v)
      print(f"Minutes to preprocess truth_values of validation data: {(time.time() - start) / 60}")
      start = time.time()

      print("Backtracker input shape", x.shape)
      
      scaling_option = 1

      if scaling_option == 1:
          (scaled_encoder,scaled_encoder_val), scaler = backtrack_scaleDataTuple((x, x_val), fit_scaler=True) 
          (scaled_decoder,scaled_decoder_val), scaler_y = backtrack_scaleDataTuple((y, y_val), fit_scaler=True) 
      else:
          n_records = x.shape[0]
          n_width = x.shape[1]
          n_sensors = int((-5. + np.sqrt(25. + 4. * np.real(n_width))) / 2. + 1.e-5)

          print('option=2: n_records,n_width,n_sensors=', n_records, n_width, n_sensors)

          scaling_factors = np.zeros(shape=12)
          scaling_factors = scalings_bjt(x, y, speed, L_scale, H_scale, n_records, n_width, n_sensors)

          scaled_encoder = scaler_bjt_x(x, scaling_factors)
          scaled_encoder_val = scaler_bjt_x(x_val, scaling_factors)
          scaled_decoder = scaler_bjt_y(y, scaling_factors)
          scaled_decoder_val = scaler_bjt_y(y_val, scaling_factors)
      
      print(f"Minutes to fit-scaler and transform-data: {(time.time() - start) / 60}")

      start = time.time()
      #If the scalers are not already saved to file (i.e. they were read from file as inputs), save them
      if not scalers_saved:
         save_scaler(scaler, join(out_path, f"bt_input_scaler.json"))
         save_scaler(scaler_y, join(out_path, f"bt_output_scaler.json"))
         scalers_saved = True
      with tf.device('/CPU:0'):
         scaled_encoder = tf.constant(scaled_encoder)
         scaled_decoder = tf.constant(scaled_decoder)
         scaled_encoder_val = tf.constant(scaled_encoder_val)
         scaled_decoder_val = tf.constant(scaled_decoder_val)
   
      x=(scaled_encoder, scaled_decoder)
      x_val = (scaled_encoder_val, scaled_decoder_val)

    if any(s in model_name for s in ("block","transformer","gaussian")) and (not preproc_done):
      start = time.time()
      encoder_data, decoder_data, leak_location, leak_rate = p.load_data(training, **config["data_options"])

      print("Train Encoder shape", encoder_data.shape)
      print(f"Minutes to load training data: {(time.time() - start) / 60}")
      start = time.time()

      scaled_encoder, scaled_decoder, encoder_mask, decoder_mask = p.preprocess(encoder_data, decoder_data,
                                                                                fit_scaler=fit_scaler)

      encoder_data_val, decoder_data_val, leak_location_val, leak_rate_val = p.load_data(validation, **config["data_options"])
      scaled_encoder_val, scaled_decoder_val, encoder_mask_val, decoder_mask_val = p.preprocess(encoder_data_val,
                                                                                                decoder_data_val,
                                                                                                fit_scaler=False)
      print("Val Encoder shape", encoder_data_val.shape)
      print(f"Minutes to fit-scaler and transform-data: {(time.time() - start) / 60}")

      start = time.time()

      #If the scalers are not already saved to file (i.e. they were read from file as inputs), save them
      if not scalers_saved:
         p.save_scalers(out_path)
         scalers_saved = True

      with tf.device('/CPU:0'):
          scaled_encoder = tf.constant(scaled_encoder)
          scaled_decoder = tf.constant(scaled_decoder)
          encoder_mask = tf.constant(encoder_mask)
          decoder_mask = tf.constant(decoder_mask)
          scaled_encoder_val = tf.constant(scaled_encoder_val)
          scaled_decoder_val = tf.constant(scaled_decoder_val)
          encoder_mask_val = tf.constant(encoder_mask_val)
          decoder_mask_val = tf.constant(decoder_mask_val)

      x=(scaled_encoder, scaled_decoder, encoder_mask, decoder_mask)
      x_val = (scaled_encoder_val, scaled_decoder_val, encoder_mask_val, decoder_mask_val)
      preproc_done = True

#for model_name in config["models"]:
    start = time.time()
    if model_name == "block_transformer_leak_loc":
        model = BlockTransformer(**config[model_name]["kwargs"])
        y, y_val = leak_location, leak_location_val
    elif model_name == "loc_rate_block_transformer":
        model = LocalizedLeakRateBlockTransformer(**config[model_name]["kwargs"])
        y = (leak_location, leak_rate)
        y_val = (leak_location_val, leak_rate_val)
    elif model_name == 'block_rate_encoder':
        model = BlockEncoder(**config[model_name]["kwargs"])
        y, y_val = leak_rate, leak_rate_val
    elif model_name == "gaussian_process":
        model = GPModel(**config[model_name]["kwargs"])
        y, y_val = leak_location, leak_location_val
    elif model_name == "backtracker":
        model = BackTrackerDNN(**config[model_name]["kwargs"])
        y_truth = y
        y = scaled_decoder
        y_truth_val = y_val
        y_val = scaled_decoder_val
    else:
        raise ValueError(f"Incompatible model type {model_name}")

    callbacks = []
    for cb in config[model_name]["callbacks"]:
        if cb == "loc_rate_metrics":
            callbacks.append(LeakLocRateMetricsCallback(x_val, y_val, batch_size=config["predict_batch_size"]))
        elif cb == "loc_only_metrics":
            callbacks.append(LeakLocMetricsCallback(x_val, y_val, batch_size=config["predict_batch_size"]))
        elif cb == "rate_only_metrics":
            callbacks.append(LeakRateMetricsCallback(x_val, y_val, batch_size=config["predict_batch_size"]))
        elif cb == "reduce_on_plateau":
            callbacks.append(ReduceLROnPlateau(**config[model_name]["callback_kwargs"][cb]))
        elif cb == "model_checkpoint":
            callbacks.append(ModelCheckpoint(filepath=os.path.join(out_path,
                                                                             f"{model_name}_{date_str}.keras")))
        elif cb == "csv_logger":
            callbacks.append(CSVLogger(join(out_path, f'training_log_{model_name}.csv'), append=True))
    config[model_name]["fit"]["callbacks"] = callbacks

    if not config["restart_options"]["restart"]:

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

        if model_name == "loc_rate_block_transformer":
            total_epochs = config[model_name]["fit"]["epochs"]
            config[model_name]["fit"]["epochs"] = config[model_name]["loss_weight_change"]["shift_epoch"]
            fit_hist = model.fit(x=x,
                                 y=y,
                                 validation_data=None,
                                 **config[model_name]["fit"])
            print(model.summary())
            config[model_name]["compile"]["loss_weights"] = config[model_name]["loss_weight_change"]["loss_weights"]
            model.compile(optimizer=optimizer, **config[model_name]["compile"])
            config[model_name]["fit"]["epochs"] = total_epochs
            fit_hist_shift = model.fit(x=x,
                                       y=y,
                                       validation_data=None,
                                       initial_epoch=config[model_name]["loss_weight_change"]["shift_epoch"],
                                       **config[model_name]["fit"])
        else:
            fit_hist = model.fit(x=x,
                                 y=y,
                                 validation_data=None,
                                 **config[model_name]["fit"])

    elif config["restart_options"]["restart"]:
        log = pd.read_csv(join(out_path, f"training_log_{model_name}.csv"))
        start_epoch = log['epoch'].iloc[-1]
        del model
        model = models.load_model(glob.glob(join(out_path, f"{model_name}*.keras"))[-1])
        fit_hist = model.fit(x=x,
                             y=y,
                             validation_data=None,
                             initial_epoch=start_epoch,
                             **config[model_name]["fit"])

    print(f"Minutes to train {model_name} model: {(time.time() - start) / 60}")
    output = model.predict(x=x_val,
                           batch_size=config["predict_batch_size"])
    output_train = model.predict(x=x,
                                 batch_size=config["predict_batch_size"])

    if model_name == "backtracker":

        if scaling_option == 1:
            (output_unscaled, output_train_unscaled, unscaled_decoder_val) = backtrack_unscaleDataTuple((output, output_train, scaled_decoder_val.numpy()), scaler_y)  

        else:

            output_unscaled = scaler_bjt_y_inverse(output, scaling_factors)
            output_train_unscaled = scaler_bjt_y_inverse(output_train, scaling_factors)
            unscaled_decoder_val = scaler_bjt_y_inverse(scaled_decoder_val.numpy(), scaling_factors)

        print('\n scaled decoder = \n', scaled_decoder.numpy(), '\n output_train_unscaled=\n', output_train_unscaled,
              '\ny_truth\n', y_truth)
        print('\n output(val,scaled)= \n', output[0:10, :],
              '\n unscl_output_val=\n', output_unscaled[0:10, :], '\n y_truth_val= \n', y_truth_val[0:10, :])

        # one more post-process step. find closest pot. leak/structure to predicted location
        output_train_unscaled = mapPredlocsToClosestPL(output_train_unscaled, x_pot_leaks, y_pot_leaks, z_pot_leaks)
        output_unscaled = mapPredlocsToClosestPL(output_unscaled, x_pot_leaks_val, y_pot_leaks_val, z_pot_leaks_val)

        print('modified output for x,y - shift to nearest potential leak structure \n')
        print('\n output_train_unscaled[10:21,:]=\n',output_train_unscaled[10:21,:],'\ny_truth[10:21,:]\n',y_truth[10:21,:])
        print('\n unscl_output_val[10:21,:]=\n',output_unscaled[10:21,:],'\n y_truth_val= \n',y_truth_val[10:21,:])

        backtracker_targets = create_binary_preds_relative(v, output_unscaled)
        pd.DataFrame(y_truth, columns=['x', 'y', 'z', 'leakrate']).to_csv(
            os.path.join(out_path, 'seals_train_true.csv'),
            index=False)
        pd.DataFrame(output_train_unscaled, columns=['x', 'y', 'z', 'leakrate']).to_csv(
            os.path.join(out_path, 'seals_train_preds.csv'),
            index=False)
        pd.DataFrame(y_truth_val, columns=['x', 'y', 'z', 'leakrate']).to_csv(
            os.path.join(out_path, 'seals_val_true.csv'),
            index=False)
        pd.DataFrame(output_unscaled, columns=['x', 'y', 'z', 'leakrate']).to_csv(
            os.path.join(out_path, 'seals_val_preds.csv'),
            index=False)

    if config["save_model"]:
        if model_name != "gaussian_process":
            model.save(os.path.join(out_path, f"{model_name}_{date_str}.keras"))

    if config["save_output"]:
        if model_name != "backtracker":
            save_output(out_path=os.path.join(out_path, f"{model_name}_output_{date_str}.nc"),
                        train_targets=y,
                        val_targets=y_val,
                        train_predictions=output_train,
                        val_predictions=output,
                        model_name=model_name)
    print('completed.')

