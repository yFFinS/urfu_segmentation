#!/bin/sh

source ../.venv/bin/activate

NODE_PARAMS="-p debug -t 00:30:00"

sbatch -n1 \
--cpus-per-task=5 \
--mem=45000 \
$NODE_PARAMS \
--job-name=mmsegm-water-debug \
--wrap="python ./train.py ./config_debug.py"
