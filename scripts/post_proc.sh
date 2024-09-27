#!/bin/bash -l
#PBS -N seals_pp
#PBS -A naml0001
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -q main
#PBS -l select=1:ncpus=16:ngpus=1:mem=84GB -l gpu_type=a100
#PBS -m a
#PBS -M cbecker@ucar.edu
module load conda
conda activate seals_202406
export TF_GPU_ALLOCATOR=cuda_malloc_async
python -u merge_eval.py