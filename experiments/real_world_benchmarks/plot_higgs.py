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
            "eval/real_world_benchmarks/higgs_rffnet_0_relevances.csv",
        ).reshape(-1, 1)
    )
).ravel()

features = np.array(
    [
        r"$\ell p_{\mathrm T}$",
        r"$\ell \eta$",
        r"$\ell \varphi$",
        "MEM",
        r"ME\,$\varphi$",
        r"Jet\,1\,$p_{\mathrm T}$",
        r"Jet\,1\,$\eta$",
        r"Jet\,1\,$\varphi$",
        r"Jet\,1\,$b$-tag",
        r"Jet\,2\,$p_{\mathrm T}$",
        r"Jet\,2\,$\eta$",
        r"Jet\,2\,$\varphi$",
        r"Jet\,2\,$b$-tag",
        r"Jet\,3\,$p_{\mathrm T}$",
        r"Jet\,3\,$\eta$",
        r"Jet\,3\,$\varphi$",
        r"Jet\,3\,$b$-tag",
        r"Jet\,4\,$p_{\mathrm T}$",
        r"Jet\,4\,$\eta$",
        r"Jet\,4\,$\varphi$",
        r"Jet\,4\,$b$-tag",
        r"$m_{jj}$",
        r"$m_{jjj}$",
        r"$m_{\ell \nu}$",
        r"$m_{j\ell \nu}$",
        r"$m_{bb}$",
        r"$m_{W bb}$",
        r"$m_{WW bb}$",
    ],
    dtype="object",
)

ordered_relevances = np.argsort(relevances)
n_features = 7

lower = ordered_relevances[:n_features]
higher = ordered_relevances[-n_features:]

x = np.append(features[lower], features[higher])
y = np.append(relevances[lower], relevances[higher])

fig, ax = plt.subplots(1, 1, frameon=False)
ax.hlines(
    np.array(range(len(features[higher]))) + n_features,
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
fig.supylabel("HIGGS features", x=4e-2, y=0.6, fontweight="bold")


lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, bbox_to_anchor=(1.07, 0.9), frameon=False)


plt.tight_layout()
plt.savefig("figures/higgs_relevances.pdf", bbox_inches="tight")
plt.show()
