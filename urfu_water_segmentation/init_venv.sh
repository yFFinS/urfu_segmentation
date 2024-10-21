#!/bin/bash
python3.9 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install poetry==1.8.3

poetry install