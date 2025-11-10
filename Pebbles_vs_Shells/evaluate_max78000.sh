#!/bin/sh
# Pebbles vs Shells evaluation script

MODEL="ai85cdnet"
DATASET="pebbles_vs_shells"
QUANTIZED_MODEL="../ai8x-synthesis/trained/ai85-pebblesshells-qat8-q.pth.tar"

python train.py --model $MODEL --dataset $DATASET --confusion --evaluate \
--exp-load-weights-from $QUANTIZED_MODEL -8 --save-sample 1 --device MAX78000 "$@"

# Evaluate script for KWS (commented out)
# MODEL="ai85kws20netv3"
# DATASET="KWS_20"
# QUANTIZED_MODEL="../ai8x-training/logs/2023.04.06-172201_kws/qat_best-q.pth.tar"
# python train.py --model $MODEL --dataset $DATASET --confusion --evaluate --exp-load-weights-from $QUANTIZED_MODEL -8 --save-sample 1 --device MAX78000 "$@"
