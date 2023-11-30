import argparse

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array

from experiments.utils import get_folder, get_generator, run_model

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
        "blobs",
        "hastie_10_2",
        "classification",
    ],
    help="Which synthetic dataset to use for the initialization experiment.",
)
parser.add_argument(
    "--model",
    "-m",
    default="rffnet",
    choices=[
        "rffnet",
        "kernel_ridge",
        "fastfood",
        "nystroem",
        "eigenpro",
        "nn",
        "nn_rff",
        "gpr",
        "srff",
        "logistic",
    ],
    help="Which synthetic dataset to use for the initialization experiment.",
)
parser.add_argument(
    "--n-samples",
    "-n",
    type=int,
    default=10_000,
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
    "--n-runs", default=20, type=int, help="Number of MC runs in the experiment."
)
parser.add_argument(
    "--max-iter",
    type=int,
    default=100,
    help="Maximum number of iterations in the optimization algorithm.",
)
parser.add_argument(
    "--batch-size",
    "-b",
    type=int,
    default=128,
    help="Batch size in the optimization algorithm.",
)
parser.add_argument(
    "--learning-rate",
    "-lr",
    type=float,
    default=1e-3,
    help="Learning rate for the optimization algorithm.",
)
parser.add_argument(
    "--n-iter-no-change",
    type=int,
    default=10,
    help="Maximum number of iterations without a decrease validation loss.",
)
parser.add_argument(
    "--memory-profile",
    "-memprof",
    action="store_true",
    help="Enable memprof decoration of estimators fit to perform memory profiling.",
)

args = parser.parse_args()

DATASET = args.dataset
N_SAMPLES = args.n_samples

MODEL = args.model

N_RANDOM_FEATURES = args.n_random_features
ALPHA = args.alpha

N_RUNS = args.n_runs

MAX_ITER = args.max_iter
BATCH_SIZE = args.batch_size
LR = args.learning_rate
N_ITER_NO_CHANGE = args.n_iter_no_change

MEM_PROF = args.memory_profile

if DATASET in ["moons", "circles", "blobs", "hastie_10_2", "classification"]:
    is_classif = True
else:
    is_classif = False

sim_benchmarks_folder = get_folder("eval/simulated_benchmarks")
generator = get_generator(DATASET)

seed_sequence = np.random.SeedSequence(entropy=0)
seeds = seed_sequence.generate_state(N_RUNS)

scaler = StandardScaler()


for i, seed in enumerate(seeds):
    X, y = generator(n_samples=N_SAMPLES + 4000, random_state=seed)
    X, X_val, y, y_val = train_test_split(X, y, test_size=2000, random_state=seed)
    X, X_test, y, y_test = train_test_split(X, y, test_size=2000, random_state=seed)

    type_y = np.int64 if is_classif else np.float32

    X = check_array(X, ensure_2d=True, dtype=np.float32)
    y = check_array(y, ensure_2d=False, dtype=type_y)
    X_val = check_array(X_val, ensure_2d=True, dtype=np.float32)
    y_val = check_array(y_val, ensure_2d=False, dtype=type_y)
    X_test = check_array(X_test, ensure_2d=True, dtype=np.float32)
    y_test = check_array(y_test, ensure_2d=False, dtype=type_y)

    X = scaler.fit_transform(X)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    if MEM_PROF:
        run = profile(run_model)
    else:
        run = run_model

    y_pred, relevances, fit_time, metrics = run(
        X, y, X_val, y_val, X_test, y_test, MODEL, is_classif, seed, args
    )

    preds_df = pd.DataFrame(columns=["preds", "target"])
    preds_df["preds"] = y_pred.flatten()
    preds_df["target"] = y_test
    preds_df.to_csv(f"{sim_benchmarks_folder}/{DATASET}_{MODEL}_{i}_preds.csv")

    if relevances is not None:
        np.savetxt(
            f"{sim_benchmarks_folder}/{DATASET}_{MODEL}_{i}_relevances.csv",
            relevances,
        )

    np.savetxt(f"{sim_benchmarks_folder}/{DATASET}_{MODEL}_{i}_metrics.csv", [metrics])
    np.savetxt(f"{sim_benchmarks_folder}/{DATASET}_{MODEL}_{i}_fittime.csv", [fit_time])
