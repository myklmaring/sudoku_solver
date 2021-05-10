#!/bin/bash

python digit_recognition.py  --datapath '/home/michael/Documents/datasets/' \
                             --test-model 'model/mnist_model_epoch_15.pth' \
                             --epochs 15 \
                             --lr 0.0001 \
                             --batch-size 128
