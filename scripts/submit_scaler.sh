#!/bin/bash -l
#PBS -N seals
#PBS -A naml0001
#PBS -l walltime=01:00:00
#PBS -o seals_train_der.out
#PBS -e seals_train_der.out
#PBS -q casper
#PBS -l select=1:ngpus=1:mem=300GB -l gpu_type=a100
#PBS -m a
#PBS -M dgagne@ucar.edu
module load conda
conda activate seals_20240515
echo $LD_LIBRARY_PATH
python -u distributed_scaling.py  -c ../config/train_transformer.yaml
