#!/bin/sh

source ../.venv/bin/activate

GPUS=2
NODE_PARAMS="-p v100 --gres=gpu:$GPUS"
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

sbatch -n1 \
--cpus-per-task=8 \
--mem=45000 \
$NODE_PARAMS \
-t 20:00:00 \
--job-name=mmsegm-water \
--wrap="python -m torch.distributed.launch \
    --nnodes=$GPUS \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=1 \
    --master_port=$PORT \
    ./train.py ./config_landcover.py \
    --launcher pytorch"
