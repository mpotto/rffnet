import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

DATASETS = ["gse1", "gse2", "jse2", "jse3"]
N_SAMPLES = [1000, 5000, 10000]

plt.style.use("rffnet.mplstyle")

fig, (ax1, ax2) = plt.subplots(2, 1, frameon=False, layout="constrained")

colors = list(mcolors.TABLEAU_COLORS.values())
transp = [x + "90" for x in colors]


def bars(ax, origin, color, n, dataset, size):
    X = origin + np.arange(n)
    metrics = np.load(f"eval/selection/{dataset.lower()}_{int(size)}.npy")
    fdr, tdr = metrics[:, 0], metrics[:, 1]
    ax.bar(
        X,
        [fdr.mean(), tdr.mean()],
        width=1.0,
        align="edge",
        color=[transp[origin // (n + 2)], colors[origin // (n + 2)]],
        yerr=[fdr.std(), tdr.std()],
    )
    # ax.boxplot([fdr, tdr])
    ax.plot([origin - 0.5, origin + n + 0.5], [0, 0], color="black", lw=2.5)
    ax.text(origin + 0.5, -0.1, "FDR", va="top", ha="center")
    ax.text(origin + n / 2 + 0.5, -0.1, "TDR", va="top", ha="center")
    size = size.replace("_", r" ")
    ax.set_ylim(0, 100)
    ax.text(origin + n / 2, -0.3, f"$n$ = {size}", va="top", ha="center")


n = 2
bars(ax1, 0 * (n + 2), "red", n, "gse1", "5_000")
bars(ax1, 1 * (n + 2), "orange", n, "gse1", "10_000")
bars(ax1, 2 * (n + 2), "teal", n, "gse1", "20_000")
ax1.text(-2.5, 0.5, "gse1", va="bottom", ha="center")

bars(ax2, 0 * (n + 2), "red", n, "gse2", "5_000")
bars(ax2, 1 * (n + 2), "orange", n, "gse2", "10_000")
bars(ax2, 2 * (n + 2), "teal", n, "gse2", "20_000")
ax2.text(-2.5, 0.5, "gse2", va="bottom", ha="center")

for ax in [ax1, ax2]:
    ax.axhline(0.5, 0.05, 1, color="0.5", linewidth=0.5, linestyle="--", zorder=-10)
    ax.axhline(1, 0.05, 1, color="0.5", linewidth=0.5, linestyle="--", zorder=-10)
    ax.axhline(0.05, 0.05, 1, color="0.5", linewidth=0.5, linestyle="--", zorder=-10)
    ax.set_xlim(-1, 3 * (n + 2) - 1.5)
    ax.set_xticks([])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.05, 0.5, 1.0])
    ax.set_yticklabels(["5%", "50%", "100%"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

plt.tight_layout()
plt.savefig("figures/selection.pdf", bbox_inches="tight")
plt.show()
