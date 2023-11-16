#!/bin/bash -l
#PBS -N seals
#PBS -A NAML0001
#PBS -l walltime=1:00:00
#PBS -o seals_train.out
#PBS -e seals_train.out
#PBS -q casper
#PBS -l select=1:ncpus=12:ngpus=1:mem=64GB -l gpu_type=v100
#PBS -m a
#PBS -M cbecker@ucar.edu
conda activate seals
cd /glade/work/cbecker/seals-ml/
python -u ./scripts/train.py  -c ./config/train_transformer.yaml