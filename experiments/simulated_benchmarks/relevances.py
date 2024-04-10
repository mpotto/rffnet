import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

import numpy as np

from experiments.utils import get_xticks

DATASETS = ["gse1", "gse2", "jse2", "jse3"]

plt.style.use("rffnet.mplstyle")

nrows, ncols = 1, 4
fig, axs = plt.subplots(
    ncols=ncols, nrows=nrows, sharey=True, layout="constrained", figsize=(5.4, 2)
)
print(axs.shape)

colors = list(mcolors.TABLEAU_COLORS.values())
color_iterator = iter(colors)


def plot(ax, dataset):
    x_ticks, x_labels, x_lim = get_xticks(dataset)
    relevances = np.abs(
        np.loadtxt(f"eval/simulated_benchmarks/{dataset}_rffnet_1_relevances.csv")
    )
    c = next(color_iterator)
    ax.bar(
        np.arange(1, len(relevances) + 1),
        relevances,
        color=c,
        width=0.8,
        zorder=10,
    )
    ax.set_title(dataset)
    ax.set_xticks(x_ticks, x_labels, fontsize=8)
    ax.set_xlim(*x_lim)
    ax.set_xlabel("Feature index")
    ax.set_ylabel(r"Relevance ($\theta$)")


for i in range(ncols):
    plot(axs[i], DATASETS[i])
    axs[i].grid(alpha=0.2, color="slategray")
    axs[i].label_outer()

plt.savefig("figures/simulated_relevances.pdf")

plt.show()
