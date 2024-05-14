#!/bin/bash -l
#PBS -N seals
#PBS -A nral0033
#PBS -l walltime=00:30:00
#PBS -o seals_train_bjt.out
#PBS -e seals_train_bjt.out
#PBS -q main
#PBS -l select=1:ncpus=1:mem=4GB -l gpu_type=a100
#PBS -m a
#PBS -M btravis@psi.edu
module load conda
conda activate seals
echo $LD_LIBRARY_PATH
cd /glade/u/home/bryant/work/temp-seals/
python -u ./scripts/train.py  -c ./config/train_transformer.yaml
