#!/bin/sh

# Training Script for Cats vs Dogs
python train.py --epochs 5 --optimizer Adam --lr 0.001 --wd 0 --deterministic \
--compress policies/schedule-catsdogs.yaml --model ai85cdnet --dataset cats_vs_dogs \
--confusion --param-hist --embedding --device MAX78000 --workers 0 --enable-tensorboard
