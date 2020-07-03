#!/bin/bash
# usage hw7_train.sh data_dir out_model
wget https://github.com/EasonLin536/ml_assignment/releases/download/0.0.2/teacher_resnet18.bin
python3 src/hw7_knowledge_distillation.py $1
python3 src/hw7_weight_quantization.py $2