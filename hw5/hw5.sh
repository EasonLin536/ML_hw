#!/bin/bash
wget https://github.com/EasonLin536/ml_assignment/releases/download/0.0.0/model_best.model
python3 src/hw5_xai.py $1 $2
