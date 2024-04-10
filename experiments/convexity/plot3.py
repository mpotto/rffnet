import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import check_random_state

plt.style.use("rffnet.mplstyle")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n-samples",
    "-n",
    type=int,
    default=500,
    help="Number of training samples in the dataset.",
)
parser.add_argument(
    "--n-random-features",
    "-s",
    type=int,
    default=300,
    help="Number of random Fourier features in the RFFNet model.",
)
parser.add_argument("--random-state", type=int, default=0)

args = parser.parse_args()

N_SAMPLES = args.n_samples
N_FEATURES = 2
N_RANDOM_FEATURES = args.n_random_features
SEED = args.random_state

rng = check_random_state(SEED)

X_test = rng.standard_normal((N_SAMPLES, N_FEATURES))

# Regression function
X = rng.standard_normal((1, 2))
alpha = rng.standard_normal((1, 1))


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


# RFF
omega = rng.standard_normal((N_FEATURES, N_RANDOM_FEATURES))
b = rng.uniform(low=0, high=2 * np.pi, size=N_RANDOM_FEATURES)


def z(x, w):
    return np.sqrt(2 / N_RANDOM_FEATURES) * np.cos((x * w) @ omega + b)


def f(w_1, w_2, beta):
    return z(X_test, np.array([w_1, w_2])) @ beta


fig, (ax1, ax2) = plt.subplots(ncols=2, layout="constrained", sharey=True)

# Plot
w_star = np.array([1, 1])
f_star = kernel_matrix(X_test, X, w_star) @ alpha

n_plot = 100
w_1 = np.linspace(-5, 5, n_plot)
w_2 = np.linspace(-5, 5, n_plot)
W_1, W_2 = np.meshgrid(w_1, w_2)
Z = np.zeros((n_plot, n_plot))
for i in range(n_plot):
    for j in range(n_plot):
        # z_vec = z(X_test, (w_1[i], w_2[j]))
        # am = z_vec.T.dot(z_vec)
        # beta = np.linalg.inv(am).dot(z_vec.T).dot(f_star)
        beta = z(X, (w_1[i], w_2[j])).T @ alpha + 0.2

        Z[i, j] = np.mean(np.square(f_star - f(w_1[i], w_2[j], beta)))

print(np.log10(Z).min(), np.log10(Z).max())
p1 = ax1.pcolor(
    W_1,
    W_2,
    np.log10(Z),
    cmap="RdBu_r",
    zorder=0,
)

plt.show()
