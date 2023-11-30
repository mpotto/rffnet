import argparse

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--classif", action="store_true", help="Plots for classification data."
)
args = parser.parse_args()

if args.classif:
    DATASETS = ["classification", "moons"]
    TAG = "classification"
else:
    DATASETS = ["gse1", "gse2", "jse2", "jse3"]
    TAG = "regression"

N_SAMPLES = [1_000, 5_000, 10_000]

plt.style.use("rffnet.mplstyle")

ncols = len(N_SAMPLES)
nrows = len(DATASETS)
fig, axs = plt.subplots(
    ncols=ncols, nrows=nrows, sharex="col", sharey="row", layout="constrained"
)

colors = list(mcolors.TABLEAU_COLORS.values())


def plot_histories(ax, dataset, n_samples):
    for i, initialization in enumerate(["constant", "restarter", "regressor"]):
        histories = np.load(
            f"eval/initialization/{dataset}_{n_samples}_{initialization}.npy"
        )
        mean = histories.mean(axis=1)
        std = histories.std(axis=1)
        epochs = np.arange(histories.shape[0])
        ax.plot(epochs, mean, label=initialization.capitalize(), lw=1, zorder=i)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.1, zorder=i)
        ax.set_xlim(0, 30)
        if dataset == "gse1":
            ax.set_ylim(0.065, 0.11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{dataset}")


# Histories
for i, dataset in enumerate(DATASETS):
    for j, n_samples in enumerate(N_SAMPLES):
        plot_histories(axs[i, j], dataset, n_samples)
        axs[0, j].set_title(r"$\mathsf{n = %d}$" % n_samples)
        axs[i, j].grid(alpha=0.2, color="slategray")
        axs[i, j].label_outer()

fig.legend(
    loc="outside upper center",
    ncol=3,
    handles=[
        Line2D([0], [0], lw=2, color=colors[0], label="Constant"),
        Line2D([0], [0], lw=2, color=colors[1], label="Restarter"),
        Line2D([0], [0], lw=2, color=colors[2], label="Regressor"),
    ],
    frameon=False,
)
fig.align_ylabels()

plt.savefig("figures/initialization_classification.pdf")
plt.show()
