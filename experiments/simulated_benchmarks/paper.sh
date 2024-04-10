#!/bin/bash

DATASETS_REGRESSION=("gse1" "gse2" "jse2" "jse3")
MODELS_REGRESSION=("rffnet" "kernel_ridge" "fastfood" "nystroem" "eigenpro" "srff" "gpr")

DATASETS_CLASSIFICATION=("classification" "moons")
MODELS_CLASSIFICATION=("rffnet" "fastfood" "nystroem" "eigenpro")

EXPERIMENT="experiments/simulated_benchmarks/generate.py"

for dataset in "${DATASETS_REGRESSION[@]}"; do
    for model in "${MODELS_REGRESSION[@]}"; do
        filename="eval/simulated_benchmarks/${dataset}_${model}_memory.dat"
        mprof run --output "$filename" --python "$EXPERIMENT" --dataset "$dataset" --model "$model" --memory-profile
    done
done


for dataset in "${DATASETS_CLASSIFICATION[@]}"; do
    for model in "${MODELS_CLASSIFICATION[@]}"; do
        filename="eval/simulated_benchmarks/${dataset}_${model}_memory.dat"
        mprof run --output "$filename" --python "$EXPERIMENT" --dataset "$dataset" --model "$model" --memory-profile
    done
done

