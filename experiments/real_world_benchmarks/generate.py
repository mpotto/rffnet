import argparse

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.validation import check_array


from experiments.utils import get_folder, run_model

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    "-d",
    default="",
    choices=[
        "amazon",
        "higgs",
        "compact",
        "powerplant",
        "abalone",
        "ailerons",
        "cpusmall",
        "yearprediction",
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
    "--n-random-features",
    type=int,
    default=300,
    help="Number of random Fourier features in the RFFNet model.",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.1,
    help="Regularization strength for the L2 penalty on the expansion weights.",
)
parser.add_argument("--n-splits", default=10, type=int, help="Number of CV splits.")
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
MODEL = args.model

N_RANDOM_FEATURES = args.n_random_features
ALPHA = args.alpha

N_SPLITS = args.n_splits

MAX_ITER = args.max_iter
BATCH_SIZE = args.batch_size
LR = args.learning_rate
N_ITER_NO_CHANGE = args.n_iter_no_change

MEM_PROF = args.memory_profile

if DATASET in ["amazon", "higgs"]:
    is_classif = True
else:
    is_classif = False

sim_benchmarks_folder = get_folder("eval/real_world_benchmarks")
data = pd.read_csv(f"data/processed/{DATASET}.csv")

X_full = data.drop(["target"], axis=1).to_numpy()
y_full = np.ravel(data[["target"]])

seed_sequence = np.random.SeedSequence(entropy=0)
seeds = seed_sequence.generate_state(N_SPLITS)

scaler = StandardScaler()

for i, (train_index, test_index) in enumerate(
    KFold(n_splits=N_SPLITS, random_state=0, shuffle=True).split(X_full)
):
    X = X_full[train_index]
    y = y_full[train_index]
    X_test = X_full[test_index]
    y_test = y_full[test_index]

    type_y = np.int64 if is_classif else np.float32

    X = check_array(X, ensure_2d=True, dtype=np.float32)
    y = check_array(y, ensure_2d=False, dtype=type_y)
    X_test = check_array(X_test, ensure_2d=True, dtype=np.float32)
    y_test = check_array(y_test, ensure_2d=False, dtype=type_y)

    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    if MODEL == "srff":
        X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, random_state=0)
        X_val = check_array(X_val, ensure_2d=True, dtype=np.float32)
        y_val = check_array(y_val, ensure_2d=False, dtype=type_y)
        X_val = scaler.transform(X_val)
    else:
        X_val, y_val = None, None

    if MEM_PROF:
        run = profile(run_model)
    else:
        run = run_model

    y_pred, relevances, fit_time, metrics = run(
        X, y, X_val, y_val, X_test, y_test, MODEL, is_classif, seeds[i], args
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

    print(metrics, fit_time)
