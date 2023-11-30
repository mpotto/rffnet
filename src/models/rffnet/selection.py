import copy

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class SelectTopK:
    def __init__(self, estimator):
        self.estimator = estimator

    def apply(self, k):
        relevances = self.estimator.relevances_

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_relevances = scaler.fit_transform(
            np.abs(relevances).reshape(-1, 1)
        ).flatten()

        mask = np.argsort(scaled_relevances)[:-k]

        estimator = copy.deepcopy(self.estimator)
        with torch.no_grad():
            estimator.module.rff.relevances.data[mask] = 0.0

        estimator.relevances_ = estimator.module.get_relevances()

        return estimator

    def path(self, X, y, score, ks=None):
        if ks is None:
            ks = list(range(X.shape[1]))

        relevances = self.estimator.relevances_

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_relevances = scaler.fit_transform(
            np.abs(relevances).reshape(-1, 1)
        ).flatten()

        argsort_relevances = np.argsort(scaled_relevances)

        scores = np.zeros(len(ks))
        estimators = []

        for k in ks:
            estimator = copy.deepcopy(self.estimator)
            mask = argsort_relevances[:-k]

            with torch.no_grad():
                estimator.module.rff.relevances.data[mask] = 0.0
            estimator.relevances_ = estimator.module.get_relevances()

            scores[k] = score(estimator.predict(X), y).item()
            estimators.append(estimator)

        results = ks, estimators, scores
        return results
