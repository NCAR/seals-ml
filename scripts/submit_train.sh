#!/bin/bash -l
#PBS -N seals
#PBS -A naml0001
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -q casper
#PBS -l select=1:ncpus=8:ngpus=1:mem=200GB -l gpu_type=a100
#PBS -m a
#PBS -M dgagne@ucar.edu
module load conda
conda activate seals_20240515
cd ../
python -u ./scripts/train.py  -c ./config/train_transformer.yaml
