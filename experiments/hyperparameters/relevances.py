import argparse

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

import numpy as np

from experiments.utils import get_xticks

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="gse1")
parser.add_argument("--n-samples", "-n", default=5000)
args = parser.parse_args()

DATASET = args.dataset
N_SAMPLES = args.n_samples

plt.style.use("rffnet.mplstyle")

alphas = [[100, 10, 1], [0.1, 0.01, 0.001]]
nrows, ncols = 2, 3
fig, axs = plt.subplots(
    ncols=ncols, nrows=nrows, sharex="col", sharey="row", layout="constrained"
)

colors = list(mcolors.TABLEAU_COLORS.values())
color_iterator = iter(colors)

x_ticks, x_labels, x_lim = get_xticks(DATASET)


def plot(ax, alpha):
    relevances = np.abs(
        np.load(f"eval/hyperparameters/{DATASET}_{N_SAMPLES}_{alpha}_relevances.npy")
    )
    mean = relevances.mean(axis=1)
    c = next(color_iterator)
    ax.bar(
        np.arange(1, len(relevances) + 1),
        mean,
        yerr=relevances.std(axis=1),
        color=c,
        width=0.8,
        ecolor="gray",
        error_kw={"elinewidth": 1},
        capsize=1,
    )
    ax.set_xticks(x_ticks, x_labels)
    ax.set_xlim(*x_lim)
    ax.set_xlabel("Feature index")


for i in range(nrows):
    for j in range(ncols):
        plot(axs[i, j], alphas[i][j])
        axs[i, j].grid(alpha=0.2, color="slategray")
        axs[i, j].label_outer()


alphas = [100, 20, 5, 1, 0.1, 0.01]
fig.legend(
    loc="outside upper center",
    ncol=len(alphas),
    handles=[
        Line2D([0], [0], lw=2, color=colors[i], label=alphas[i])
        for i in range(len(alphas))
    ],
    frameon=False,
)
fig.supylabel("Relevances")
#  plt.savefig(f"figures/alpha_{DATASET}_{N_SAMPLES}.pdf")
plt.show()
