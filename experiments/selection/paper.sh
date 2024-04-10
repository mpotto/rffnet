#!/bin/bash

DATASETS=("gse1" "gse2" "jse2" "jse3")
N_SAMPLES=(5000 10000 20000)

BATCH_SIZES=(256 512 512)

EXPERIMENT="experiments/selection/generate.py"

i=0

for n_samples in "${N_SAMPLES[@]}"; do
    bs=${BATCH_SIZES[i]}
    for dataset in "${DATASETS[@]}"; do
        python "$EXPERIMENT" --dataset "$dataset" --n-samples "$n_samples" --batch-size "$bs" -lr 0.001
    done
    i=$((i+1))
done
