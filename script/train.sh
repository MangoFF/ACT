#!/bin/bash
#SBATCH --job-name=multi_node_dpo
#SBATCH --output=./output/multi_node_dpo.out
#SBATCH --error=./output/multi_node_dpo.err
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --gres=gpu:8
#SBATCH --exclusive 
#SBATCH --exclude g0066,g0036,g0067,g0029,g0021,g0017,g0011
#SBATCH --nodes=2
echo "START TIME: $(date)"

# 每个节点的显卡数
GPUS_PER_NODE=8

# 主节点和IP
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID 
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CMD="act/train.py \
    --train_args_file config/dpo.json"

LAUNCHER="accelerate launch \
    --multi_gpu \
    --num_machines $NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port $MASTER_PORT \
    --num_processes $WORLD_SIZE \
    --machine_rank \$SLURM_PROCID \
    --role \$SLURMD_NODENAME: \
    --rdzv_conf rdzv_backend=c10d \
    --max_restarts 0 \
    --tee 3 \
"

SRUN_ARGS=" \
    --wait=120 \
    --kill-on-bad-exit=1 \
"

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER $CMD" 

echo "END TIME: $(date)"

