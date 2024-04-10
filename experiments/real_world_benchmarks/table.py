import argparse
import glob
import tempfile

import mprof
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--classification", action="store_true")
args = parser.parse_args()

IS_CLASSIF = args.classification

if IS_CLASSIF:
    DATASETS = ["amazon", "higgs", "a9a", "w8a"]
    MODELS = ["rffnet", "eigenpro", "fastfood", "nystroem"]
else:
    DATASETS = ["compact", "abalone", "yearprediction", "powerplant"]
    MODELS = ["rffnet", "eigenpro", "fastfood", "nystroem", "gpr", "srff", "kernel_ridge"]

real_benchmarks_folder = "eval/real_world_benchmarks"

df = pd.DataFrame()


for dataset in DATASETS:
    for model in MODELS:
        if dataset in ["amazon", "higgs"] and model in ["eigenpro"]:
            pass
        elif dataset == "yearprediction" and model in ["eigenpro", "kernel_ridge", "gpr"]:
            pass
        else:
            metrics = []
            fit_times = []
            n_runs = len(
                glob.glob(f"{real_benchmarks_folder}/{dataset}_{model}_*_metrics.csv")
            )
            for i in range(n_runs):
                # Metrics
                if IS_CLASSIF:
                    metrics.append(
                        np.loadtxt(
                            f"{real_benchmarks_folder}/{dataset}_{model}_{i}_metrics.csv"
                        )[-1]
                    )
                else:
                    metrics.append(
                        np.loadtxt(
                            f"{real_benchmarks_folder}/{dataset}_{model}_{i}_metrics.csv"
                        )
                    )

                # Fit times
                fit_times.append(
                    np.loadtxt(
                        f"{real_benchmarks_folder}/{dataset}_{model}_{i}_fittime.csv"
                    )
                )

            # Memory
            full_memory_profile = mprof.read_mprofile_file(
                f"{real_benchmarks_folder}/{dataset}_{model}_memory.dat"
            )
            try:
                timestamps = full_memory_profile["func_timestamp"][
                    "experiments.utils.run_model"
                ]
            except KeyError:
                print(
                    "Precise timestamps were not created. Printing the peak memory instead."
                )
                # memory = np.max(full_memory_profile["mem_usage"])
            else:
                memory = np.array([np.mean(t[-3:-1]) for t in timestamps])
            # Concat results
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            [
                                dataset,
                                model,
                                np.mean(metrics),
                                np.std(metrics),
                                np.mean(fit_times),
                                np.std(fit_times),
                                np.mean(memory),
                                np.std(memory),
                            ]
                        ]
                    ),
                ],
            )

df.columns = [
    "dataset",
    "Model",
    "metric_mean",
    "metric_std",
    "fit_time_mean",
    "fit_time_std",
    "memory_mean",
    "memory_std",
]

df = df.round(3)

for col in ["metric", "fit_time", "memory"]:
    df["result_" + col] = (
        df[col + "_mean"]
        .astype(str)
        .str.ljust(4, "0")
        .str.cat(df[col + "_std"].astype(str).str.ljust(4, "0"), sep=" \pm ")
    )
    df["result_" + col] = "$" + df["result_" + col] + "$"

    df["dataset"] = df["dataset"].str.replace("_", "\_")
    df = df.drop([col + "_mean", col + "_std"], axis=1)

df["Model"] = df["Model"].replace(
    {
        "rffnet": "RFFNet",
        "eigenpro": "EigenPro",
        "nystroem": "Nyström",
        "fastfood": "Fastfood",
        "nn": "Neural Net",
        "srff": "SRFF",
        "kernel_ridge": "Kernel Ridge",
        "logistic": "Logistic",
    }
)

df = df.rename({"dataset": "Dataset"}, axis=1)

# Metrics table
df1 = df.pivot(index="Model", columns="Dataset", values="result_metric")
df1 = df1.reset_index().rename_axis(None, axis=1)

if IS_CLASSIF:
    name = "classification"
else:
    name = "regression"

df1.to_latex(
    f"tables/{name}-real-metrics",
    index=False,
    escape=False,
    column_format="l" * df.shape[1],
)

# Memory table
df2 = df.pivot(index="Model", columns="Dataset", values="result_memory")
df2 = df2.reset_index().rename_axis(None, axis=1)

df2.to_latex(
    f"tables/{name}-real-memory",
    index=False,
    escape=False,
    column_format="l" * df.shape[1],
)

# Fit time
df3 = df.pivot(index="Model", columns="Dataset", values="result_fit_time")
df3 = df3.reset_index().rename_axis(None, axis=1)

df3.to_latex(
    f"tables/{name}-real-fittime",
    index=False,
    escape=False,
    column_format="l" * df.shape[1],
)
