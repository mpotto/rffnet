import copy

import numpy as np
import torch

from sklearn.utils import check_random_state


class Constant:
    def __init__(self, scale=None):
        self.scale = scale

    def initialize(self, X, y, module, solver, datafit, penalty):
        if self.scale:
            scale = self.scale
        else:
            scale = X.shape[1]

        rel_init = (
            torch.max(X, axis=0).values - torch.min(X, axis=0).values
        ).float() / scale

        return rel_init


class Restarter:
    def __init__(
        self,
        n_restarts=10,
        max_iter=100,
        warm_restart=True,
        scale=None,
        random_state=None,
    ):
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.warm_restart = warm_restart
        self.scale = scale
        self.random_state = random_state

    def initialize(self, X, y, module, solver, datafit, penalty):
        rng = check_random_state(self.random_state)

        module = copy.deepcopy(module)
        initial_module = copy.deepcopy(module)

        solver = copy.deepcopy(solver)
        solver.max_iter = self.max_iter

        n_features = X.shape[1]

        if self.scale:
            scale = self.scale
        else:
            scale = n_features

        for i in range(self.n_restarts):
            rel_init = torch.from_numpy(
                rng.normal(
                    loc=(torch.max(X, axis=0).values - torch.min(X, axis=0).values)
                    / scale,
                    scale=(1 / scale),
                    size=(n_features),
                )
            ).float()

            best_rel = rel_init
            best_loss = np.inf

            with torch.no_grad():
                module.rff.relevances.data = rel_init

            solver.solve(
                X,
                y,
                module,
                datafit,
                penalty,
            )

            if solver.best_loss < best_loss:
                best_loss = solver.best_loss
                best_rel = rel_init

            if not self.warm_restart:
                module = initial_module

            print(best_rel)

        return best_rel


class Regressor:
    def __init__(self, regressor):
        self.regressor = regressor

    def initialize(self, X, y, module, criterion, solver, penalty) -> torch.Tensor:
        self.regressor.fit(X, y)
        return torch.from_numpy(self.regressor.coef_).float()
