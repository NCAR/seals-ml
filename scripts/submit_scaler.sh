#!/bin/bash -l
#PBS -N seals_scaler
#PBS -A naml0001
#PBS -l walltime=02:00:00
#PBS -o seals_train_der.out
#PBS -e seals_train_der.out
#PBS -q casper
#PBS -l select=1:ncpus=36:ngpus=0:mem=300GB
#PBS -m a
#PBS -M dgagne@ucar.edu
module load conda
conda activate seals_20240515
python -u distributed_scaling.py  -c ../config/train_transformer.yaml
