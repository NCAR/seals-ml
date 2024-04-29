#!/bin/bash -l
#PBS -N seals
#PBS -A NAML0001
#PBS -l walltime=00:20:00
#PBS -o seals_data_gen.out
#PBS -e seals_data_gen.out
##PBS -q main
#PBS -q casper
#PBS -l select=1:ncpus=12:mem=12GB
#PBS -m a
#PBS -M btravis@psi.edu
module load conda
conda activate seals
cd ~/work/temp-seals/
python -u ./scripts/generate_data.py  -c ./config/generate_training_data_bjt.yaml
