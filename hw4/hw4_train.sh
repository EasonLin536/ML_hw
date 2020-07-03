#!/bin/bash
wget https://github.com/EasonLin536/ml_assignment/releases/download/0.0.1/w2v_all.model
wget https://github.com/EasonLin536/ml_assignment/releases/download/0.0.1/w2v_all.model.trainables.syn1neg.npy
wget https://github.com/EasonLin536/ml_assignment/releases/download/0.0.1/w2v_all.model.wv.vectors.npy
python3 src/hw4_train.py $1 $2 $3