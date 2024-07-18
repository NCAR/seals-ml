import argparse
from sealsml.data import Preprocessor
from bridgescaler import save_scaler
from sklearn.model_selection import train_test_split
from glob import glob
from multiprocessing import Pool
import yaml
import numpy as np
from os.path import join
import datetime
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file")
    parser.add_argument("-p", "--procs", type=int, default=1, help="Number of CPUs to use")

    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    np.random.seed(config["random_seed"])
    username = os.environ.get('USER')
    config["out_path"] = config["out_path"].replace("username", username)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    out_path = os.path.join(config["out_path"], f"scalers_{date_str}")

    os.makedirs(out_path, exist_ok=False)
    with open(join(out_path, 'train.yml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    files = sorted(glob(os.path.join(config["data_path"], "*.nc")))
    training, validation = train_test_split(files,
                                            test_size=config["validation_ratio"],
                                            random_state=config["random_seed"])
    p = Preprocessor(scaler_type=config["scaler_type"], sensor_pad_value=-1, sensor_type_value=-999)
    p.save_filenames(training, validation, out_path)
    pool = Pool(processes=args.procs)
    all_scalers = pool.map(fit_scaler_single_file, training)
    all_scalers_arr = np.array(all_scalers)
    total_scalers_arr = np.sum(all_scalers_arr, axis=0)
    save_scaler(total_scalers_arr[0], join(out_path, "coord_scaler.json"))
    save_scaler(total_scalers_arr[0], join(out_path, "sensor_scaler.json"))
    return


def fit_scaler_single_file(nc_file, scaler_type, scaler_options):
    print(nc_file)
    p = Preprocessor(scaler_type=scaler_type, sensor_pad_value=-1, sensor_type_value=-999,
                     scaler_options=scaler_options)
    encoder_data, decoder_data, leak_location, leak_rate = p.load_data([nc_file])
    scaled_encoder, scaled_decoder, encoder_mask, decoder_mask = p.preprocess(encoder_data, decoder_data,
                                                                              fit_scaler=True)
    return p.coord_scaler, p.sensor_scaler


if __name__ == "__main__":
    main()