#!/bin/bash -l
#PBS -N seals
#PBS -A NAML0001
#PBS -l walltime=03:00:00
#PBS -o seals_data_gen.out
#PBS -e seals_data_gen.out
#PBS -q main
#PBS -l select=1:ncpus=120:mem=128GB
#PBS -m a
#PBS -M cbecker@ucar.edu
module load conda
conda activate seals
cd /glade/work/cbecker/seals-ml/
python -u ./scripts/generate_data.py  -c ./config/generate_training_data.yaml
