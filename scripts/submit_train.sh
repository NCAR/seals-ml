#!/bin/bash -l
#PBS -N seals
#PBS -A nral0033
#PBS -l walltime=4:00:00
#PBS -o seals_train_der.out
#PBS -e seals_train_der.out
#PBS -q main
#PBS -l select=1:ngpus=4:mem=128GB -l gpu_type=a100
#PBS -m a
#PBS -M dgagne@ucar.edu
module load conda
conda activate seals
export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}

export CUDNN_PATH=${CONDA_PREFIX}
export LD_LIBRARY_PATH=${CUDNN_PATH}/lib/:${TENSORRT_PATH}:${LD_LIBRARY_PATH}

export XLA_FLAGS=--xla_gpu_cuda_data_dir=${CONDA_PREFIX}

# For CUDA Aware MPI support
export OMPI_MCA_opal_cuda_support=true

# For TF optimizations, see https://github.com/NVIDIA/DeepLearningExamples/issues/57
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2 # 2 (default) or sometimes 4 seems a little better
python -u train.py  -c ../config/train_transformer.yaml
