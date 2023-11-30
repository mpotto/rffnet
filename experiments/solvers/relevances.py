import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

import numpy as np

from experiments.utils import get_xticks

DATASET = "gse2"
N_SAMPLES = 5000

plt.style.use("rffnet.mplstyle")

solvers = ["sgd", "adam", "palm-sgd", "palm-adam"]
nrows, ncols = 2, 2
fig, axs = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    sharex="col",
    sharey="row",
    layout="constrained",
)

colors = list(mcolors.TABLEAU_COLORS.values())
color_iterator = iter(colors)

x_ticks, x_labels, x_lim = get_xticks(DATASET)


def plot(ax, solver):
    relevances = np.abs(
        np.load(f"eval/solvers/{DATASET}_{N_SAMPLES}_{solver}_relevances.npy")
    )
    mean = relevances.mean(axis=1)
    c = next(color_iterator)
    ax.bar(
        np.arange(1, len(relevances) + 1),
        mean,
        yerr=relevances.std(axis=1),
        color=c,
        ecolor="gray",
        error_kw={"elinewidth": 1},
        capsize=1,
    )
    ax.set_xticks(x_ticks, x_labels)
    ax.set_xlim(*x_lim)
    ax.set_xlabel("Feature index")


solvers = np.array(solvers).reshape(2, 2)
for i in range(nrows):
    for j in range(ncols):
        plot(axs[i, j], solvers[i, j])
        axs[i, j].grid(alpha=0.2, color="slategray")
        axs[i, j].label_outer()


fig.legend(
    loc="outside upper center",
    ncol=4,
    handles=[
        Line2D([0], [0], lw=2, color=colors[0], label="SGD"),
        Line2D([0], [0], lw=2, color=colors[1], label="Adam"),
        Line2D([0], [0], lw=2, color=colors[2], label="PALM-SGD"),
        Line2D([0], [0], lw=2, color=colors[3], label="PALM-Adam"),
    ],
    frameon=False,
)
fig.align_ylabels()
fig.supylabel("Relevances")

plt.savefig(f"figures/initialization_{DATASET}_{N_SAMPLES}.pdf")
plt.show()
