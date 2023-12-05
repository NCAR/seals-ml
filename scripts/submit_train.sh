#!/bin/bash -l
#PBS -N seals
#PBS -A NAML0001
#PBS -l walltime=4:00:00
#PBS -o seals_train_der.out
#PBS -e seals_train_der.out
#PBS -q main
#PBS -l select=1:ngpus=4:mem=128GB -l gpu_type=a100
#PBS -m a
#PBS -M cbecker@ucar.edu
module load conda
conda activate seals
cd /glade/work/cbecker/seals-ml/
python -u ./scripts/train.py  -c ./config/train_transformer.yaml