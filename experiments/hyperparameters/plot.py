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

if axs.ndim == 1:
    axs = axs.reshape((-1, 1))

colors = list(mcolors.TABLEAU_COLORS.values())
alphas = [100, 10, 1, 0.1, 0.01, 0.001]


def plot(ax, dataset, n_samples):
    for i, alpha in enumerate(alphas):
        histories = np.load(
            f"eval/hyperparameters/{dataset}_{n_samples}_{alpha}_history.npy"
        )
        mean = histories.mean(axis=1)
        std = histories.std(axis=1)
        epochs = np.arange(histories.shape[0])
        ax.plot(epochs, mean, lw=1, zorder=i)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.1, zorder=i)
        ax.set_xlim(0, 60)
        if dataset == "gse1":
            ax.set_ylim(0.065, 0.11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{dataset}")


for i, dataset in enumerate(DATASETS):
    for j, n_samples in enumerate(N_SAMPLES):
        axs[0, j].set_title(r"$\mathsf{n = %d}$" % n_samples)
        plot(axs[i, j], dataset, n_samples)
        axs[i, j].grid(alpha=0.2, color="slategray")
        axs[i, j].label_outer()

fig.legend(
    loc="outside upper center",
    ncol=len(alphas),
    handles=[
        Line2D([0], [0], lw=2, color=colors[i], label=alphas[i])
        for i in range(len(alphas))
    ],
    frameon=False,
)
fig.align_ylabels()
plt.savefig(f"figures/alpha_{TAG}.pdf")
plt.show()
