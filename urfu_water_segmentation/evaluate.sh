#!/bin/sh

source .venv/bin/activate

# Для запуска на CPU, за 2 часа успевает
NODE_PARAMS="-p apollo -t 06:00:00"
DEVICE="cpu"

# Для запуска на GPU, успевает за 10-15 минут
# NODE_PARAMS="-p hiperf --gres=gpu:a100:1 --nodelist=tesla-a101 -t 03:00:00"
# NODE_PARAMS="-p hiperf --gres=gpu:v100:1 --nodelist=tesla-v100 -t 03:00:00"
# DEVICE="cuda"

EXPERIMENT_PATH="./landcover/logs/Poolformer_LandcoverAI_512_FocalLoss_AdamW_bsize_128"

sbatch -n1 \
    --cpus-per-task=8 \
    --mem=45000 \
    $NODE_PARAMS \
    --job-name="mmsegm-water-eval" \
    --wrap="python ./evaluate.py --output-path=./eval_\${SLURM_JOB_ID}.csv --experiment-path=$EXPERIMENT_PATH --device=$DEVICE"
