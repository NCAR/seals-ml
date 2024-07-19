#!/bin/bash -l
#PBS -N seals_train
#PBS -A naml0001
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -q main
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -m a
#PBS -M dgagne@ucar.edu
module load conda
conda activate seals_20240515
cd ../
python -u ./scripts/train_multi.py  -c ./config/train_transformer.yaml
