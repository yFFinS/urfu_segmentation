#!/bin/sh

source .venv/bin/activate

# Папка с картинками для визуализации
IMAGES_PATH=/misc/home1/m_imm_freedata/Segmentation/Projects/mmseg_water/landcover.ai_512/val/images
MASKS_PATH=/misc/home1/m_imm_freedata/Segmentation/Projects/mmseg_water/landcover.ai_512/val/gt

# Для запуска на CPU
NODE_PARAMS="-p apollo -t 06:00:00"
DEVICE="cpu"
BATCH_SIZE=1

# Для запуска на GPU
# NODE_PARAMS="-p hiperf --gres=gpu:a100:1 --nodelist=tesla-a101 -t 03:00:00"
# NODE_PARAMS="-p hiperf --gres=gpu:v100:1 --nodelist=tesla-v100 -t 03:00:00"
# DEVICE="cuda"
# BATCH_SIZE=8

EXPERIMENT_PATH="./landcover/logs/Poolformer_LandcoverAI_512_FocalLoss_AdamW_bsize_128"

sbatch -n1 \
    --cpus-per-task=8 \
    --mem=45000 \
    $NODE_PARAMS \
    --job-name="mmsegm-water-visualize" \
    --wrap="python ./visualization.py \
            --experiment-path=$EXPERIMENT_PATH \
            --images-path=$IMAGES_PATH \
            --masks-path=$MASKS_PATH \
            --output-path=./visualized_\${SLURM_JOB_ID} \
            --device=$DEVICE \
            --batch-size=$BATCH_SIZE"
