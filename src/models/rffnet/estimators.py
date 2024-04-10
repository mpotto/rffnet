import warnings
import time

import numpy as np
import torch
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
)
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, check_consistent_length
from sklearn.preprocessing import LabelEncoder
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import Adam

from src.models.rffnet.initialization import Restarter, Constant
from src.models.rffnet.module import RFFNet
from src.models.rffnet.penalties import L2, Null
from src.models.rffnet.solvers import SingleBlock


def _rffnet_fit(X, y, model, datafit, initializer, penalty, solver):
    is_classif = isinstance(datafit, CrossEntropyLoss)

    if y.ndim == 2 and y.shape[1] == 1:
        warnings.warn(
            "DataConversionWarning('A column-vector y"
            " was passed when a 1d array was expected"
        )
        y = y[:, 0]

    if is_classif:
        enc = LabelEncoder()
        enc.fit(y)
        model.classes_ = enc.classes_
        n_classes = len(enc.classes_)
        y = check_array(y, ensure_2d=False, dtype=np.int64)
        y = torch.from_numpy(y)
    else:
        y = check_array(y, ensure_2d=False, dtype=np.float32)
        y = torch.from_numpy(y.reshape(-1, 1))

    X = check_array(X, accept_sparse=False, dtype=np.float32, ensure_2d=True)
    X = torch.from_numpy(X)

    check_consistent_length(X, y)

    n_features = X.shape[1]
    if is_classif:
        module = RFFNet(
            dims=(n_features, model.n_random_features, n_classes), sampler=model.sampler
        )
    else:
        module = RFFNet(
            dims=(n_features, model.n_random_features, 1), sampler=model.sampler
        )

    if initializer:
        relevances_init = initializer.initialize(X, y, module, solver, datafit, penalty)

        with torch.no_grad():
            module.rff.relevances.data = relevances_init

    t_start = time.time()
    model.module = solver.solve(X, y, module, datafit, penalty)
    fit_time = time.time() - t_start

    model.module.eval()
    model.relevances_ = module.get_relevances()
    model.coefs_ = module.linear.weight.detach().numpy()
    model.fit_time = fit_time

    return model


class RFFNetEstimator(BaseEstimator):
    def __init__(
        self,
        n_random_features,
        sampler=torch.randn,
        datafit=None,
        initializer=None,
        penalty=None,
        solver=None,
    ):
        self.n_random_features = n_random_features
        self.sampler = sampler
        self.datafit = datafit
        self.initializer = initializer
        self.penalty = penalty
        self.solver = solver

    def fit(self, X, y):
        return _rffnet_fit(
            X, y, self, self.datafit, self.initializer, self.penalty, self.solver
        )

    def predict(self, X):
        return self._decision_function(X).numpy()

    def _decision_function(self, X):
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse=False, dtype=np.float32, ensure_2d=True)
        X = torch.from_numpy(X)
        return self.module(X).detach()


class RFFNetRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_random_features=300,
        sampler=torch.randn,
        alpha=1e-4,
        n_restarts=0,
        max_init_iter=10,
        warm_restart=True,
        max_iter=100,
        batch_size=32,
        lr=1e-3,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        checkpoint=True,
        verbose=False,
        random_state=None,
    ):
        self.n_random_features = n_random_features
        self.sampler = sampler
        self.alpha = alpha
        self.n_restarts = n_restarts
        self.max_init_iter = max_init_iter
        self.warm_restart = warm_restart
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lr = lr
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.checkpoint = checkpoint
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y):
        if self.n_restarts > 0:
            initializer = Restarter(
                n_restarts=self.n_restarts,
                max_iter=self.max_init_iter,
                warm_restart=self.warm_restart,
                random_state=self.random_state,
            )
        else:
            initializer = Constant()
        solver = SingleBlock(
            Adam,
            batch_size=self.batch_size,
            lr=self.lr,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change,
            validation_fraction=self.validation_fraction,
            checkpoint=self.checkpoint,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        penalty = (L2(self.alpha), Null())

        return _rffnet_fit(X, y, self, MSELoss(), initializer, penalty, solver)

    def predict(self, X):
        return self._decision_function(X).numpy()

    def _decision_function(self, X):
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse=False, dtype=np.float32, ensure_2d=True)
        X = torch.from_numpy(X)
        return self.module(X).detach()


class RFFNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_random_features=300,
        sampler=torch.randn,
        alpha=1e-4,
        n_restarts=0,
        max_init_iter=10,
        warm_restart=True,
        max_iter=100,
        batch_size=32,
        lr=1e-3,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        checkpoint=True,
        verbose=False,
        random_state=None,
    ):
        self.n_random_features = n_random_features
        self.sampler = sampler
        self.alpha = alpha
        self.n_restarts = n_restarts
        self.max_init_iter = max_init_iter
        self.max_iter = max_iter
        self.warm_restart = warm_restart
        self.batch_size = batch_size
        self.lr = lr
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.checkpoint = checkpoint
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y):
        if self.n_restarts > 0:
            initializer = Restarter(
                n_restarts=self.n_restarts,
                max_iter=self.max_init_iter,
                warm_restart=self.warm_restart,
                random_state=self.random_state,
            )
        else:
            initializer = Constant()
        solver = SingleBlock(
            Adam,
            batch_size=self.batch_size,
            lr=self.lr,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change,
            validation_fraction=self.validation_fraction,
            checkpoint=self.checkpoint,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        penalty = (L2(self.alpha), Null())

        return _rffnet_fit(X, y, self, CrossEntropyLoss(), initializer, penalty, solver)

    def predict(self, X):
        scores = self._decision_function(X)
        return scores.argmax(dim=1).numpy()

    def predict_proba(self, X):
        return torch.softmax(self._decision_function(X), -1).numpy()

    def _decision_function(self, X):
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse=False, dtype=np.float32, ensure_2d=True)
        X = torch.from_numpy(X)
        return self.module(X).detach()
