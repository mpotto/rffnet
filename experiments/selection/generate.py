import argparse

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.preprocessing import StandardScaler

from experiments.utils import get_folder, get_generator, get_support
from src.models.rffnet.estimators import RFFNetEstimator
from src.models.rffnet.initialization import Constant
from src.models.rffnet.penalties import L2, Null
from src.models.rffnet.solvers import SingleBlock
from src.models.rffnet.selection import SelectTopK

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    "-d",
    type=str,
    default="gse1",
    choices=["gse1", "gse2", "jse2", "jse3"],
    help="Dataset for the selection experiment."
)
parser.add_argument(
    "--n-samples",
    "-n",
    type=int,
    default=1000,
    help="Number of training samples in the dataset."
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
    default=1,
    help="Regularization strength for the L2 penalty on the expansion weights.",
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
    default=64,
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
    "--n-iter-no-change",
    type=int,
    default=10,
    help="Maximum number of iterations without a decrease validation loss.",
)

args = parser.parse_args()

N_SAMPLES = args.n_samples
DATASET = args.dataset

N_RANDOM_FEATURES = args.n_random_features
ALPHA = args.alpha

N_RUNS = args.n_runs

MAX_ITER = args.max_iter
BATCH_SIZE = args.batch_size
LR = args.learning_rate
N_ITER_NO_CHANGE = args.n_iter_no_change

selection_folder = get_folder("eval/selection")
generator = get_generator(DATASET)
support = get_support(DATASET)

seed_sequence = np.random.SeedSequence(entropy=0)
seeds = seed_sequence.generate_state(N_RUNS)

datafit = torch.nn.MSELoss()
penalty = (L2(ALPHA), Null())
initializer = Constant()

scaler = StandardScaler()

results = np.zeros((N_RUNS, 2))

for i, seed in enumerate(seeds):
    X, y = generator(
        n_samples=N_SAMPLES + 2_000,
        random_state=seed,
    )
    X, X_s, y, y_s = train_test_split(X, y, test_size=2_000, random_state=seed)

    X = scaler.fit_transform(X)
    X_s = scaler.transform(X_s)

    torch.manual_seed(seed)

    solver = SingleBlock(
        batch_size=BATCH_SIZE,
        lr=LR,
        max_iter=MAX_ITER,
        early_stopping=True,
        n_iter_no_change=N_ITER_NO_CHANGE,
        verbose=False,
        random_state=seed,
    )

    model = RFFNetEstimator(
        n_random_features=N_RANDOM_FEATURES,
        datafit=datafit,
        initializer=initializer,
        penalty=penalty,
        solver=solver,
    )

    model.fit(X, y)

    selector = SelectTopK(model)
    ks, estimators, scores = selector.path(X_s, y_s, mean_squared_error)

    best_k = np.argsort(scores)[0]
    best_estimator = estimators[best_k]
    best_relevances = best_estimator.relevances_

    nz_relevances = np.nonzero(best_relevances)

    support_mask = np.zeros(best_relevances.shape[0], dtype=np.int32)
    relevances_mask = np.zeros(best_relevances.shape[0], dtype=np.int32)

    support_mask[support] = 1
    relevances_mask[nz_relevances] = 1

    matrix = confusion_matrix(support_mask, relevances_mask, labels=[0, 1])
    fpr = matrix[1, 0] / matrix[:, 0].sum()
    tpr = matrix[1, 1] / matrix[:, 1].sum()

    results[i, 0] = fpr
    results[i, 1] = tpr
    

np.save(
    f"{selection_folder}/{DATASET}_{N_SAMPLES}.npy", results
)
