#!/bin/bash

DATASETS=("jse2" "jse3" "gse1" "gse2" "classification" "moons")
N_SAMPLES=(1000 5000 10000)

LRS=(0.001 0.001 0.001 0.001 0.01 0.01)
BATCH_SIZES=(64 256 256)
RANDOM_FEATURES=(100 300 300)

EXPERIMENT="experiments/solvers/generate.py"

i=0
for n_samples in "${N_SAMPLES[@]}"; do
    
    rf="${RANDOM_FEATURES[i]}"
    bs="${BATCH_SIZES[i]}"
    
    j=0
    for dataset in "${DATASETS[@]}"; do
        lr="${LRS[j]}"
        python "$EXPERIMENT" --dataset "$dataset" --n-samples "$n_samples" -lr "$lr" --batch-size "$bs" --n-random-features "$rf"
        j=$((j+1))   
    done
    i=$((i+1))
done

