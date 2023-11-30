import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.preprocessing import MinMaxScaler

plt.style.use("rffnet.mplstyle")

colors = list(mcolors.TABLEAU_COLORS.values())

scaler = MinMaxScaler()
relevances = scaler.fit_transform(
    np.abs(
        np.loadtxt(
            "eval/real_world_benchmarks/amazon_rffnet_0_relevances.csv",
        ).reshape(-1, 1)
    )
).ravel()

features = np.loadtxt(
    "eval/real_world_benchmarks/amazon_rffnet_0_features.csv", dtype="object"
)

n_words = 7
ordered_relevances = np.argsort(relevances)

lower = ordered_relevances[:n_words]
higher = ordered_relevances[-n_words:]

x = np.append(features[lower], features[higher])
y = np.append(relevances[lower], relevances[higher])

fig, ax = plt.subplots(1, 1, frameon=False)
ax.hlines(
    np.array(range(len(features[higher]))) + n_words,
    0,
    relevances[higher],
    color="tab:red",
    alpha=0.8,
)
ax.hlines(
    np.array(range(len(features[lower]))),
    0,
    relevances[lower],
    color="tab:blue",
    alpha=0.8,
)
ax.set_yticks(np.array(range(len(x))), x)
ax.set_xscale("log")

ax.set_xlabel(r"Relevance ($\theta$)")
ax.set_xlim(1e-6, 1.1)
fig.supylabel("Stemmed words", x=4e-2, y=0.6, fontweight="bold")


lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, bbox_to_anchor=(1.07, 0.9), frameon=False)

plt.tight_layout()
plt.savefig("figures/amazon_relevances.pdf", bbox_inches="tight")
plt.show()
