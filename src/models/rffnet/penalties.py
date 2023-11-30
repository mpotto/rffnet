import numpy as np


def prox_2_squared(x, u):
    """Compute the proximal operator of the l2 norm squared."""
    return x / (1 + 2 * u)


def prox_1(x, u):
    """Entrywise soft-thresholding of array x at level u."""
    return np.sign(x) * np.maximum(0.0, np.abs(x) - u)


class L2:
    def __init__(self, alpha):
        self.alpha = alpha

    def value(self, w):
        return self.alpha * np.sum(np.square(w))

    def prox(self, w, stepsize):
        return prox_2_squared(w, self.alpha * stepsize)


class L1:
    def __init__(self, alpha):
        self.alpha = alpha

    def value(self, w):
        return self.alpha * np.sum(np.abs(w))

    def prox(self, w, stepsize):
        return prox_1(w, self.alpha * stepsize)


class Null:
    def __init__(self):
        self.alpha = 0.0

    def value(self, w):
        return 0

    def prox(self, w, stepsize):
        return w
