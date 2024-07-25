#!/bin/bash
#SBATCH --job-name=multi_node_dpo
#SBATCH --output=./output/multi_node_dpo.out
#SBATCH --error=./output/multi_node_dpo.err
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --gres=gpu:8
#SBATCH --exclusive 
#SBATCH --exclude g0066,g0036,g0067,g0029,g0021,g0017,g0011
#SBATCH --nodes=2

source $(pwd)/.env

echo $MODEL_PATH
echo $DS_CONFIG_PATH

srun bash -c 'echo $MODEL_PATH'