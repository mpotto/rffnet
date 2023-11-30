import argparse

import numpy as np
import torch
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from experiments.utils import get_folder, get_generator
from src.models.rffnet.estimators import RFFNetEstimator
from src.models.rffnet.initialization import Constant, Regressor, Restarter
from src.models.rffnet.penalties import L2, Null
from src.models.rffnet.solvers import PALM

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
        "piironen",
        "classification",
        "circles",
        "moons",
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
    default=200,
    help="Number of random Fourier features in the RFFNet model.",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=1e-4,
    help="Regularization strength for the L2 penalty on the expansion weights.",
)
parser.add_argument(
    "--n-runs", default=50, type=int, help="Number of MC runs in the experiment."
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
parser.add_argument(
    "--n-restarts",
    type=int,
    default=10,
    help="Number of restarts in the Restarter initialization scheme.",
)
parser.add_argument(
    "--max-init-iter",
    type=int,
    default=20,
    help="Maximum number of iterations in the optimization runs in the Restarter initialization scheme.",
)
parser.add_argument(
    "--warm-restart",
    action="store_true",
    help="Restart the model with previously trained random features weights in the Restarter initialization scheme.",
)

args = parser.parse_args()

DATASET = args.dataset
N_SAMPLES = args.n_samples

N_RANDOM_FEATURES = args.n_random_features
ALPHA = args.alpha

N_RUNS = args.n_runs

MAX_ITER = args.max_iter
BATCH_SIZE = args.batch_size
LR = args.learning_rate

N_RESTARTS = args.n_restarts
MAX_INIT_ITER = args.max_init_iter
WARM_RESTART = args.warm_restart

initialization_folder = get_folder("eval/initialization")
generator = get_generator(DATASET)

seed_sequence = np.random.SeedSequence(entropy=0)
seeds = seed_sequence.generate_state(N_RUNS)

if DATASET in ["classification", "circles", "moons"]:
    datafit = torch.nn.CrossEntropyLoss()
else:
    datafit = torch.nn.MSELoss()

penalty = (L2(ALPHA), Null())

scaler = StandardScaler()

for strategy in ["constant", "restarter", "regressor"]:
    results = np.zeros((MAX_ITER, N_RUNS))

    for i, seed in enumerate(seeds):
        if strategy == "constant":
            init = Constant()
        elif strategy == "restarter":
            init = Restarter(
                n_restarts=N_RESTARTS,
                max_iter=MAX_INIT_ITER,
                warm_restart=WARM_RESTART,
                random_state=seed,
            )
        elif strategy == "regressor":
            base_regressor = Lasso(random_state=seed)
            init = Regressor(base_regressor)

        X, y = generator(n_samples=N_SAMPLES + 2_000, random_state=seed)
        torch.manual_seed(seed)

        X = scaler.fit_transform(X)

        solver = PALM(
            batch_size=BATCH_SIZE,
            lr=LR,
            max_iter=MAX_ITER,
            validation_fraction=2_000,
            early_stopping=False,
            verbose=True,
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

    np.save(f"{initialization_folder}/{DATASET}_{N_SAMPLES}_{strategy}.npy", results)
