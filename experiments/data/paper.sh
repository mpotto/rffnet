#!/bin/bash

# Download datasets 
python experiments/data/download.py -d compact
python experiments/data/download.py -d abalone
python experiments/data/download.py -d yearprediction
python experiments/data/download.py -d powerplant
python experiments/data/download.py -d a9a
python experiments/data/download.py -d w8a
python experiments/data/download.py -d higgs
python experiments/data/download.py -d amazon

# Process all datasets
python experiments/data/process.py -d compact
python experiments/data/process.py -d abalone
python experiments/data/process.py -d yearprediction
python experiments/data/process.py -d powerplant
python experiments/data/process.py -d a9a
python experiments/data/process.py -d w8a
python experiments/data/process.py -d higgs
python experiments/data/process.py -d amazon
