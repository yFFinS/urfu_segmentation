#!/bin/bash
python3 -m venv .venv # create venv
source ./.venv/bin/activate # activate venv
pip3 install torch torchvision torchaudio # torch
pip install -U openmim # install mim for install some mm libs
mim install mmengine # mmengine from mim
mim install "mmcv>=2.0.0" # mmcv from mim
pip install -e ../ # install mmseg
pip install ftfy regex # some libs to start train
pip install future tensorboard # tensorboard
