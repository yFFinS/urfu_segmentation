#!/bin/sh

source ../.venv/bin/activate

NODE_PARAMS="-p v100 --gres=gpu:1"

sbatch -n1 \
--cpus-per-task=8 \
--mem=45000 \
$NODE_PARAMS \
-t 20:00:00 \
--job-name=mmsegm-water \
--wrap="python ./train.py ./config_landcover.py"
