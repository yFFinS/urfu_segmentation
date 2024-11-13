#!/bin/sh

source .venv/bin/activate

# Select best node params
NODE_PARAMS="-p apollo -t 20:00:00"  # might not finish in time
DEVICE="cpu"

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
