#!/bin/bash -l
#PBS -N seals_scaler
#PBS -A naml0001
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -q main
#PBS -l select=1:ncpus=127:ngpus=0
#PBS -m a
#PBS -M dgagne@ucar.edu
module load conda
conda activate seals_20240515
python -u distributed_scaling.py  -c ../config/train_transformer.yaml -p 120
