import copy

import numpy as np
import torch
from sklearn.model_selection import train_test_split


class PALM:
    def __init__(
        self,
        optimizer=torch.optim.Adam,
        batch_size=32,
        lr=1e-2,
        max_iter=100,
        early_stopping=False,
        n_iter_no_change=10,
        validation_fraction=0.1,
        checkpoint=True,
        random_state=None,
        verbose=False,
    ):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.lr = lr
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.checkpoint = checkpoint
        self.random_state = random_state
        self.verbose = verbose

    def solve(self, X, y, module, datafit, penalty):
        optim_linear = self.optimizer(module.linear.parameters(), lr=self.lr)
        optim_relevances = self.optimizer(module.rff.parameters(), lr=self.lr)

        X, X_val, y, y_val = train_test_split(
            X, y, test_size=self.validation_fraction, random_state=self.random_state
        )

        no_improvement_count = 0
        self.best_loss = torch.from_numpy(np.array(np.inf))
        self.history = []

        best_model_state_dict = copy.deepcopy(module.state_dict())

        n_samples = len(X)
        for n_iter in range(self.max_iter):
            indices = torch.randperm(n_samples)
            module.train()

            for i in range(n_samples // self.batch_size):
                batch = indices[i * self.batch_size : (i + 1) * self.batch_size]
                pred = module(X[batch])

                loss = datafit(pred, y[batch])

                optim_linear.zero_grad()
                loss.backward()
                optim_linear.step()

                with torch.no_grad():
                    module.linear.weight.data = penalty[0].prox(
                        module.linear.weight.data, self.lr
                    )

            for i in range(n_samples // self.batch_size):
                batch = indices[i * self.batch_size : (i + 1) * self.batch_size]
                pred = module(X[batch])

                loss = datafit(pred, y[batch])

                optim_relevances.zero_grad()
                loss.backward()
                optim_relevances.step()

                with torch.no_grad():
                    module.rff.relevances.data = penalty[1].prox(
                        module.rff.relevances.data, self.lr
                    )

            with torch.no_grad():
                val_loss = datafit(module(X_val), y_val)

            if val_loss < self.best_loss:
                best_model_state_dict = copy.deepcopy(module.state_dict())
                self.best_loss = val_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count == self.n_iter_no_change and self.early_stopping:
                break

            if self.verbose:
                print(f"Epoch {n_iter}, Val. loss {val_loss:.3e}")

            self.history.append(val_loss)

        if self.checkpoint:
            module.load_state_dict(best_model_state_dict)

        return module


class SingleBlock:
    def __init__(
        self,
        optimizer=torch.optim.Adam,
        batch_size=32,
        lr=1e-2,
        max_iter=100,
        early_stopping=False,
        n_iter_no_change=10,
        validation_fraction=0.1,
        checkpoint=True,
        random_state=None,
        verbose=False,
    ):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.lr = lr
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.checkpoint = checkpoint
        self.random_state = random_state
        self.verbose = verbose

    def solve(self, X, y, module, datafit, penalty):
        optim = self.optimizer(module.parameters(), lr=self.lr)

        X, X_val, y, y_val = train_test_split(
            X, y, test_size=self.validation_fraction, random_state=self.random_state
        )

        no_improvement_count = 0
        self.best_loss = torch.from_numpy(np.array(np.inf))

        self.history = []

        best_model_state_dict = copy.deepcopy(module.state_dict())

        n_samples = len(X)
        for n_iter in range(self.max_iter):
            indices = torch.randperm(n_samples)
            module.train()

            for i in range(n_samples // self.batch_size):
                batch = indices[i * self.batch_size : (i + 1) * self.batch_size]
                pred = module(X[batch])

                loss = datafit(pred, y[batch])

                optim.zero_grad()
                loss.backward()
                optim.step()

                with torch.no_grad():
                    module.linear.weight.data = penalty[0].prox(
                        module.linear.weight.data, self.lr
                    )

            with torch.no_grad():
                val_loss = datafit(module(X_val), y_val)

            if val_loss < self.best_loss:
                best_model_state_dict = copy.deepcopy(module.state_dict())
                self.best_loss = val_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count == self.n_iter_no_change and self.early_stopping:
                break

            if self.verbose:
                print(f"Epoch {n_iter}, Val. loss {val_loss:.3e}")

            self.history.append(val_loss)

        if self.checkpoint:
            module.load_state_dict(best_model_state_dict)

        return module
