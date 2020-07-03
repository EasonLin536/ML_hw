#!/bin/bash
wget https://github.com/EasonLin536/ml_assignment/releases/download/0.0.2/model_best.bin
python3 src/hw7_test.py $1 $2