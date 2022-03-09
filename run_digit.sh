#!/bin/bash

python digit_recognition.py  --datapath '/home/michael/Documents/datasets/' \
                             --epochs 15 \
                             --lr 0.0001 \
                             --batch-size 128 \
                             --test-model 'model/mnist_model_epoch_15.pth'
