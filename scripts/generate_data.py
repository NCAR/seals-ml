import argparse
import yaml
import itertools
import os
from os.path import join
from multiprocessing import Pool
from sealsml.data import DataSampler
import glob
import datetime


def main(config, file):

    sampler = DataSampler(**config["sampler_args"])
    sampler.load_data(file)
    data = sampler.sample(time_window_size=config["time_window_size"],
                          samples_per_window=config["samples_per_window"],
                          window_stride=config["window_stride"])

    data.to_netcdf(join(config["out_path"], f"training_data_{file.split('/')[-1]}.nc"))
    del sampler


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)

    username = os.environ.get('USER')
    config["out_path"] = config["out_path"].replace("username", username)
    date_str = datetime.datetime.now().strftime("data_gen_%Y%m%d")
    config['out_path'] = os.path.join(config["out_path"], date_str)
    os.makedirs(config["out_path"], exist_ok=True)
    with open(join(config["out_path"], 'config.yml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    files = sorted(glob.glob(join(config["data_path"], "*")))
    args = itertools.product([config], files)

    if config["parallel"]:
        n_procs = int(config["n_processors"])
        with Pool(n_procs) as pool:
            pool.starmap(main, args)
    else:
        for f in files:
            main(config, f)



