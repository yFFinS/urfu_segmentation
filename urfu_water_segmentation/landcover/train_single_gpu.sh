#!/bin/sh

source ../.venv/bin/activate

NODE_PARAMS="-p hiperf --gres=gpu:a100:1 --nodelist=tesla-a101 -t 20:00:00"

sbatch -n1 \
--cpus-per-task=12 \
--mem=45000 \
$NODE_PARAMS \
--job-name=mmsegm-water \
--wrap="python ./train.py ./config_landcover.py"
