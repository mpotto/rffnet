import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import check_random_state

plt.style.use("rffnet.mplstyle")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--w-star",
    type=float,
    default=2,
    help="Number of random Fourier features in the RFFNet model.",
)
parser.add_argument("--random-state", type=int, default=0)

args = parser.parse_args()

W_STAR = args.w_star
SEED = args.random_state
rng = check_random_state(SEED)


def gaussian(x, y, w):
    return np.exp(-0.5 * np.sum(np.square(w * (x - y))))


def kernel_matrix(X, Y, w):
    n = X.shape[0]
    q = Y.shape[0]

    K = np.zeros((n, q))
    for i in range(n):
        for j in range(q):
            K[i, j] = gaussian(X[i, :], Y[j, :], w)
    return K


# Regression function
X = rng.standard_normal((1, 2))
alpha = rng.standard_normal((1, 1))


def z(x, omega, b, w_1, w_2):
    w = np.array([w_1, w_2])
    return np.sqrt(2 / omega.shape[1]) * np.cos((x * w) @ omega + b)


def f(X_test, omega, b, w_1, w_2, beta):
    w = np.array([w_1, w_2])
    z = lambda x: np.sqrt(2 / omega.shape[1]) * np.cos((x * w) @ omega + b)
    return z(X_test) @ beta


fig, (ax1, ax2) = plt.subplots(ncols=2, layout="constrained", sharey=True)

# Plot 1
n_samples1 = 200
s1 = 100

omega1 = rng.standard_normal((2, s1))
b1 = rng.uniform(low=0, high=2 * np.pi, size=s1)

X_test = rng.standard_normal((n_samples1, 2))
f_star = kernel_matrix(X_test, X, W_STAR) @ alpha

n_plot = 50
w_1 = np.linspace(-10, 10, n_plot)
w_2 = np.linspace(-10, 10, n_plot)
W_1, W_2 = np.meshgrid(w_1, w_2)
Z = np.zeros((n_plot, n_plot))
for i in range(n_plot):
    for j in range(n_plot):
        beta = z(X, omega1, b1, w_1[i], w_2[j]).T @ alpha
        Z[i, j] = np.mean(np.square(f_star - f(X_test, omega1, b1, w_1[i], w_2[j], beta)))


p1 = ax1.pcolor(W_1, W_2, np.log10(Z), cmap="RdBu_r", zorder=0, vmin=-3, vmax=0.0)

ax1.set_xlabel(r"$\theta_1$", fontsize=12)
ax1.set_ylabel(r"$\theta_2$", fontsize=12)
ax1.set_aspect("equal", "box")

ax1.scatter(W_STAR, W_STAR, color="k", s=5, zorder=10)
ax1.scatter(W_STAR, -W_STAR, color="k", s=5, zorder=10)
ax1.scatter(-W_STAR, W_STAR, color="k", s=5, zorder=10)
ax1.scatter(-W_STAR, -W_STAR, color="k", s=5, zorder=10)

# Plot 1
n_samples2 = 1000
s2 = 500

omega2 = rng.standard_normal((2, s2))
b2 = rng.uniform(low=0, high=2 * np.pi, size=s2)

X_test = rng.standard_normal((n_samples2, 2))
f_star = kernel_matrix(X_test, X, W_STAR) @ alpha

n_plot = 50
w_1 = np.linspace(-10, 10, n_plot)
w_2 = np.linspace(-10, 10, n_plot)
W_1, W_2 = np.meshgrid(w_1, w_2)
Z = np.zeros((n_plot, n_plot))
for i in range(n_plot):
    for j in range(n_plot):
        beta = z(X, omega2, b2, w_1[i], w_2[j]).T @ alpha
        Z[i, j] = np.mean(np.square(f_star - f(X_test, omega2, b2, w_1[i], w_2[j], beta)))


p2 = ax2.pcolor(W_1, W_2, np.log10(Z), cmap="RdBu_r", zorder=0, vmin=-3, vmax=0.0)

ax2.set_xlabel(r"$\theta_1$", fontsize=12)
ax2.set_aspect("equal", "box")
ax2.scatter(W_STAR, W_STAR, color="k", s=5, zorder=10)
ax2.scatter(W_STAR, -W_STAR, color="k", s=5, zorder=10)
ax2.scatter(-W_STAR, W_STAR, color="k", s=5, zorder=10)
ax2.scatter(-W_STAR, -W_STAR, color="k", s=5, zorder=10)

# Colorbar
cbar = fig.colorbar(p2, ax=ax2, shrink=0.5, aspect=10)
cbar.set_label("Log MSE")

plt.savefig(
    f"figures/population_{W_STAR}_samples.pdf",
    bbox_inches="tight",
)

plt.show()
