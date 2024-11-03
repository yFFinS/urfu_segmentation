#!/bin/sh

source ../.venv/bin/activate

GPUS=2
NODE_PARAMS="-p hiperf --gres=gpu:a100:$GPUS --nodelist=tesla-a101 -t 00:15:00"

sbatch -n1 \
    --cpus-per-task=8 \
    --mem=45000 \
    $NODE_PARAMS \
    --job-name=mmsegm-water \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS} \
    --wrap="python ./train.py ./config_debug.py --launcher slurm"
