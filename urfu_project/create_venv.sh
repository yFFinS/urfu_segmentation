#!/bin/bash
# Create a conda environment and activate it.
conda create --name openmmlab python=3.8 -y
conda activate openmmlab

# Install PyTorch
pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html

# Install MMCV using MIM
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# Install MMSegmentation
cd mmsegmentation
pip install -v -e .

# Install some libs to start training
pip install ftfy regex

# Install TensorBoard for visualization
pip install tensorboardX
pip install future tensorboard
