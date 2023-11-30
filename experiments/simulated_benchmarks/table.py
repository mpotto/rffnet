import argparse

import mprof
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--classification", action="store_true")
args = parser.parse_args()

IS_CLASSIF = args.classification

if IS_CLASSIF:
    DATASETS = ["classification", "moons"]
    MODELS = ["rffnet", "eigenpro", "fastfood", "nystroem"]
else:
    DATASETS = ["gse1", "gse2", "jse2", "jse3"]
    MODELS = ["rffnet", "eigenpro", "fastfood", "nystroem", "gpr", "srff", "kernel_ridge"]

N_RUNS = 10

sim_benchmarks_folder = "eval/simulated_benchmarks"

pd.set_option("display.float_format", lambda x: "%.3f" % x)

df = pd.DataFrame()


def format_tex(float_number):
    exponent = np.floor(np.log10(float_number))
    mantissa = float_number / 10**exponent
    mantissa_format = str(mantissa)[0:3]
    return "${0}\times10^{{{1}}}$".format(mantissa_format, str(int(exponent)))


for dataset in DATASETS:
    for model in MODELS:
        metrics = []
        fit_times = []

        for i in range(N_RUNS):
            # Metrics
            if IS_CLASSIF:
                metrics.append(
                    np.loadtxt(
                        f"{sim_benchmarks_folder}/{dataset}_{model}_{i}_metrics.csv"
                    )[-1]
                )
            else:
                metrics.append(
                    np.loadtxt(
                        f"{sim_benchmarks_folder}/{dataset}_{model}_{i}_metrics.csv"
                    )
                )
            # Fit times
            fit_times.append(
                np.loadtxt(f"{sim_benchmarks_folder}/{dataset}_{model}_{i}_fittime.csv")
            )

        # Memory
        full_memory_profile = mprof.read_mprofile_file(
            f"{sim_benchmarks_folder}/{dataset}_{model}_memory.dat"
        )
        timestamps = full_memory_profile["func_timestamp"]["experiments.utils.run_model"]
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

df = df.round(decimals=3)
df = df.reset_index(drop=True)

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
        "gpr": "GPR",
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
    f"tables/{name}-synthetic-metrics",
    index=False,
    escape=False,
    column_format="l" * df.shape[1],
)

# Memory table
df2 = df.pivot(index="Model", columns="Dataset", values="result_memory")
df2 = df2.reset_index().rename_axis(None, axis=1)

df2.to_latex(
    f"tables/{name}-synthetic-memory",
    index=False,
    escape=False,
    column_format="l" * df.shape[1],
)

# Fit time
df3 = df.pivot(index="Model", columns="Dataset", values="result_fit_time")
df3 = df3.reset_index().rename_axis(None, axis=1)

df3.to_latex(
    f"tables/{name}-synthetic-fittime",
    index=False,
    escape=False,
    column_format="l" * df.shape[1],
)
