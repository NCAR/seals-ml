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

    # sampler args
    sampler = DataSampler(**config["sampler_args"])

    # loading in the data with xarray and save out number of sources
    ds, num_sources = sampler.load_data(file)

    for i in range(len(num_sources)):

        sampler.data_extract(ds.isel(srcDim=i))
        
        # this would not have to change below
        data = sampler.sample(time_window_size=config["time_window_size"],
                            samples_per_window=config["samples_per_window"],
                            window_stride=config["window_stride"])

        srcDim = f"srcDim{i}"  # Construct srcDim with loop variable i
        file_name = f"training_data_{file.split('/')[-1].split('.')[0]}_{srcDim}.nc"  # Modify the file name string

        data.to_netcdf(os.path.join(config["out_path"], file_name))  # Save the DataFrame to netCDF file with the modified file name
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
    files = sorted(glob.glob(join(config["data_path"], "**", "*.*"), recursive=True))
    print(files)
    args = itertools.product([config], files)
    if config["parallel"]:
        n_procs = int(config["n_processors"])
        with Pool(n_procs) as pool:
            pool.starmap(main, args)
    else:
        for f in files:
            main(config, f)
