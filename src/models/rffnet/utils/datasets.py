import numpy as np
from sklearn.utils import check_random_state


def make_correlated_design(n_samples, n_features, rho=0.5, random_state=None):
    rng = check_random_state(random_state)

    if rho != 0:
        sigma = np.sqrt(1 - rho * rho)
        U = rng.randn(n_samples)

        X = np.empty([n_samples, n_features], order="F")
        X[:, 0] = U
        for j in range(1, n_features):
            U *= rho
            U += sigma * rng.randn(n_samples)
            X[:, j] = U
    else:
        X = rng.randn(n_samples, n_features)
    return X


# Papers
def make_correlated_data(
    n_samples=200,
    n_features=10,
    rho=0.5,
    noise_level=0.1,
    density=0.1,
    w_true=None,
    random_state=None,
):
    rng = check_random_state(random_state)

    X = make_correlated_design(n_samples, n_features, rho, random_state)

    nnz = int(density * n_features)
    if w_true is None:
        w_true = np.zeros((n_features, 1))
        support = rng.choice(n_features, nnz, replace=False)
        w_true[support, :] = rng.randn(nnz).reshape(-1, 1)
    else:
        if w_true.ndim == 1:
            w_true = w_true.reshape(-1, 1)

    y = X @ w_true + noise_level * rng.randn(n_samples, 1)
    y = y.flatten()

    return X, y, w_true


def make_jordan_se1(
    n_samples=200, n_features=10, rho=0.5, noise_level=0.1, random_state=None
):
    rng = check_random_state(random_state)

    X = make_correlated_design(n_samples, n_features, rho, random_state)

    y = X[:, 0] + noise_level * rng.randn(n_samples)
    return X, y


def make_jordan_se2(
    n_samples=300, n_features=10, rho=0.5, noise_level=0.1, random_state=None
):
    rng = check_random_state(random_state)

    X = make_correlated_design(n_samples, n_features, rho, random_state)

    y = X[:, 0] ** 3 + X[:, 1] ** 3 + noise_level * rng.randn(n_samples)
    return X, y


def make_jordan_se3(n_samples=200, n_features=10, noise_level=0.1, random_state=None):
    rng = check_random_state(random_state)

    X = rng.randn(n_samples, n_features)

    y = X[:, 0] * X[:, 1] + noise_level * rng.randn(n_samples)
    return X, y


def make_gregorova_se1(n_samples=2000, noise_level=0.1, random_state=None):
    rng = check_random_state(random_state)

    X = rng.randn(n_samples, 18)

    y = np.sin(np.square(X[:, 0] + X[:, 2])) * np.sin(
        X[:, 6] * X[:, 7] * X[:, 8]
    ) + noise_level * rng.randn(n_samples)
    return X, y


def make_gregorova_se2(n_samples=2000, noise_level=0.1, random_state=None):
    rng = check_random_state(random_state)

    X = rng.randn(n_samples, 100)

    y = np.log(np.square(np.sum(X[:, 10:15], axis=1))) + noise_level * rng.randn(
        n_samples
    )
    return X, y


def make_piironen(n_samples=300, noise_level=0.3, random_state=None):
    rng = check_random_state(random_state)

    X = -1 + 2 * rng.rand(n_samples, 8)

    phi = np.linspace(np.pi / 10, np.pi, 8)
    coefs = np.sqrt(2.0 / (1 - np.sin(phi) * np.cos(phi) / phi))

    y = np.sin(X * phi) @ coefs + noise_level * rng.randn(n_samples)
    return X, y


def make_rodeo_se1(n_samples=300, noise_level=0.5, random_state=None):
    rng = check_random_state(random_state)

    X = rng.uniform(low=0.0, high=1.0, size=(n_samples, 10))

    y = 5 * X[:, 0] ** 2 * X[:, 1] ** 2 + noise_level * rng.randn(n_samples)
    return X, y


def make_rodeo_se2(n_samples=300, noise_level=1, random_state=None):
    rng = check_random_state(random_state)

    X = rng.uniform(low=0.0, high=1.0, size=(n_samples, 20))

    y = (
        2 * (X[:, 0] + 1) ** 3
        + 2 * np.sin(10 * X[:, 1])
        + noise_level * rng.randn(n_samples)
    )

    return X, y


def make_chen_se1(n_samples=300, n_features=50, noise_level=0.1, random_state=None):
    rng = check_random_state(random_state)

    X = rng.randn(n_samples, n_features)

    y = (
        0.1 * (X[:, 0] + X[:, 1] + X[:, 2]) ** 3
        + np.tanh(X[:, 0] + X[:, 2] + X[:, 4])
        + noise_level * rng.randn(n_samples)
    )
    return X, y


def make_fully_redundant(n_samples=300, n_features=2, noise_level=0.1, random_state=None):
    rng = check_random_state(random_state)

    X = np.tile(rng.randn(n_samples, 1), (1, n_features))

    y = X[:, 0] + X[:, 1] + X[:, 2] + 5 * X[:, 3] + noise_level * rng.randn(n_samples)

    return X, y
