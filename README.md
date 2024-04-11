# RFFNet

## Setup Instructions

* Scripts must be run from the `/rffnet` folder.
* Create the `conda` environment by doing `conda env create --file environment.yml`.
* Make the methods available by doing `conda develop .`.
* Activate the environment via `conda activate rffnet_env`.
* If you want to re-run the experiment, setup folders with `experiments/setup/setup.sh`. This overrides the `figures` and `tables` folders with blank ones.

## Download and Process Datasets

* Manually download the `amazon` dataset from the link in the `experiments/data/download.py` script. 
* Run the `experiments/data/paper.sh` script. 

## Replicating the Tables and Figures

### Tables 1, F.7, F.11

* Run`experiments/simulated_benchmarks/paper.sh` and `experiments/simulated_benchmarks/table.py`.

### Tables 2, F.8, F.12

* Run `experiments/simulated_benchmarks/table.py --classication`.

### Table 3, F.9, F.13

* Run `experiments/real_world_benchmarks/paper.sh` and `experiments/real_world_benchmarks/table.py`.

### Table 4, F.10, F.14

* Run `experiments/real_world_benchmarks/table.py --classification`.

### Figure 2

* Run `experiments/simulated_benchmarks/relevances.py`.

### Figure 3

* Run `experiments/selection/paper.sh` and `experiments/selection/plot.py`.

### Figure 4

* Run `experiments/real_world_benchmarks/plot_amazon.py` and `experiments/real_world_benchmarks/plot_higgs.py`.

### Figure D.5

* Run `experiments/samplers/paper.sh` and `experiments/samplers/plot.py`.

### Figure D.6

* Run `experiments/initialization/paper.sh` and `experiments/initialization/plot.py`.

### Figure D.7

* Run `experiments/solvers/paper.sh` and `experiments/solvers/plot.py`.

### Figure E.8

* Run `experiments/hyperparameters/paper.sh` and `experiments/hyperparameters/plot.py`.

### Figure E.9 and E.10 

* Run `experiments/hyperparameters/relevances.py --gse1` and `experiments/hyperparameters/relevances.py --gse2`.

### Landscape Figure

* Run `experiments/convexity/plot.py` and `experiments/convexity/plot2.py`.

## Folder Contents

* `data`: raw data (i.e., as is from source) and processed data.
* `eval`: experiment results (metrics, training histories, relevance vectors).
* `examples`: simple scripts to play with `RFFNet`.
* `experiments`: code for experiments.
* `figures`: figures resulting from the experiments. 
* `src`: source code for `rffnet`, `srff`, and `gpr`'s with ARD kernels, as well as code for generating synthetic datasets.
* `tables`: tables with results from the experiments.