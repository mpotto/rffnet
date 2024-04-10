import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler

from experiments.utils import get_folder, get_generator
from src.models.rffnet.estimators import RFFNetEstimator
from src.models.rffnet.initialization import Constant
from src.models.rffnet.penalties import L2, Null
from src.models.rffnet.solvers import SingleBlock

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    "-d",
    default="gse1",
    choices=[
        "gse1",
        "gse2",
        "jse1",
        "jse2",
        "jse3",
        "moons",
        "circles",
        "classification",
    ],
    help="Which synthetic dataset to use for the initialization experiment.",
)
parser.add_argument(
    "--n-samples",
    "-n",
    type=int,
    default=1000,
    help="Number of training samples in the dataset.",
)
parser.add_argument(
    "--n-random-features",
    type=int,
    default=300,
    help="Number of random Fourier features in the RFFNet model.",
)
parser.add_argument(
    "--n-runs", default=10, type=int, help="Number of MC runs in the experiment."
)
parser.add_argument(
    "--max-iter",
    type=int,
    default=100,
    help="Maximum number of iterations in the PALM optimization algorithm.",
)
parser.add_argument(
    "--batch-size",
    "-b",
    type=int,
    default=128,
    help="Batch size in the PALM optimization algorithm.",
)
parser.add_argument(
    "--learning-rate",
    "-lr",
    type=float,
    default=1e-3,
    help="Learning rate for the PALM optimization algorithm.",
)

args = parser.parse_args()

DATASET = args.dataset
N_SAMPLES = args.n_samples

N_RANDOM_FEATURES = args.n_random_features

N_RUNS = args.n_runs

MAX_ITER = args.max_iter
BATCH_SIZE = args.batch_size
LR = args.learning_rate

hyperparameters_folder = get_folder("eval/hyperparameters")

generator = get_generator(DATASET)
X, _ = generator(n_samples=3)
n_features = X.shape[1]

seed_sequence = np.random.SeedSequence(entropy=0)
seeds = seed_sequence.generate_state(N_RUNS)

if DATASET in ["moons", "circles", "classification"]:
    datafit = torch.nn.CrossEntropyLoss()
else:
    datafit = torch.nn.MSELoss()
    
init = Constant()

scaler = StandardScaler()

alphas = [100, 50, 10, 1, 0.5, 1e-1, 1e-2, 1e-3, 1e-4]

for a in alphas:
    results = np.zeros((MAX_ITER, N_RUNS))
    relevances = np.zeros((n_features, N_RUNS))

    penalty = (L2(a), Null())

    for i, seed in enumerate(seeds):
        X, y = generator(n_samples=N_SAMPLES + 2_000, random_state=seed)

        torch.manual_seed(seed)

        X = scaler.fit_transform(X)

        solver = SingleBlock(
            torch.optim.Adam,
            batch_size=BATCH_SIZE,
            lr=LR,
            max_iter=MAX_ITER,
            validation_fraction=2_000,
            early_stopping=False,
            verbose=False,
            random_state=seed,
        )

        model = RFFNetEstimator(
            n_random_features=N_RANDOM_FEATURES,
            datafit=datafit,
            initializer=init,
            penalty=penalty,
            solver=solver,
        )

        model.fit(X, y)

        results[:, i] = model.solver.history
        relevances[:, i] = model.relevances_

    np.save(
        f"{hyperparameters_folder}/{DATASET}_{N_SAMPLES}_{a}_history.npy",
        results,
    )
    np.save(
        f"{hyperparameters_folder}/{DATASET}_{N_SAMPLES}_{a}_relevances.npy",
        relevances,
    )
