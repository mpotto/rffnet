#!/bin/bash

DATASETS_REGRESSION=("compact" "abalone" "powerplant")
MODELS_REGRESSION=("rffnet" "kernel_ridge" "fastfood" "nystroem" "eigenpro" "gpr" "srff")

DATASETS_BIG_REGRESSION=("yearprediction")
MODELS_BIG_REGRESSION=("rffnet")  #"fastfood" "nystroem" "srff")

DATASETS_CLASSIFICATION=("a9a" "w8a")
MODELS_CLASSIFICATION=("rffnet" "fastfood" "nystroem" "eigenpro")

DATASETS_BIG_CLASSIFICATION=("amazon" "higgs")
MODELS_BIG_CLASSIFICATION=("rffnet" "fastfood" "nystroem")

EXPERIMENT="experiments/real_world_benchmarks/generate.py"

# Regression
for dataset in "${DATASETS_REGRESSION[@]}"; do
    for model in "${MODELS_REGRESSION[@]}"; do
        filename="eval/real_world_benchmarks/${dataset}_${model}_memory.dat"
        mprof run --output "$filename" --python "$EXPERIMENT" --dataset "$dataset" --model "$model" --memory-profile
    done
done


for dataset in "${DATASETS_BIG_REGRESSION[@]}"; do
    for model in "${MODELS_BIG_REGRESSION[@]}"; do
        filename="eval/real_world_benchmarks/${dataset}_${model}_memory.dat"
        mprof run --output "$filename" --python "$EXPERIMENT" --dataset "$dataset" --model "$model" -lr 1e-3 --alpha 1e-4 --n-random-features 1000 --memory-profile
    done
done


# Classification
for dataset in "${DATASETS_CLASSIFICATION[@]}"; do
    for model in "${MODELS_CLASSIFICATION[@]}"; do
        filename="eval/real_world_benchmarks/${dataset}_${model}_memory.dat"
        mprof run --output "$filename" --python "$EXPERIMENT" --dataset "$dataset" --model "$model" --memory-profile
    done
done

for dataset in "${DATASETS_BIG_CLASSIFICATION[@]}"; do
    for model in "${MODELS_BIG_CLASSIFICATION[@]}"; do
        filename="eval/real_world_benchmarks/${dataset}_${model}_memory.dat"
        mprof run --output "$filename" --python "$EXPERIMENT" --dataset "$dataset" --model "$model" -lr 1e-3 --memory-profile
    done
done
