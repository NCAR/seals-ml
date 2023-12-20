from glob import glob
import os
from os.path import exists, join
import numpy as np
import shutil
in_path = "/glade/work/cbecker/SEALS_training_data_20231205/"
out_path = "/glade/derecho/scratch/dgagne/SEALS_training_data_20231206/"
subsample = 0.1

training_files = np.array(sorted(glob(in_path + "*.nc")))
num_sub_files = int(len(training_files) * subsample)
np.random.shuffle(training_files)
training_sub = training_files[:num_sub_files]
print(training_sub.size, num_sub_files)
if not exists(out_path):
    os.makedirs(out_path)
out_links = sorted(glob(out_path + "*.nc"))
for out_link in out_links:
    os.remove(out_link)
for training_file in training_sub:
    shutil.copy(training_file, out_path + training_file.split("/")[-1])

