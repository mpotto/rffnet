import time
import os

import torch
import torch.nn as nn

from sklearn_extra.kernel_methods import EigenProClassifier, EigenProRegressor
from sklearn_extra.kernel_approximation import Fastfood
from sklearn.datasets import make_moons, make_circles, make_blobs, make_hastie_10_2, make_classification
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from skorch import NeuralNetRegressor
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, Checkpoint
from skorch.probabilistic import ExactGPRegressor

from src.models.gpr.gpr import ARDModule
from src.models.rffnet.estimators import RFFNetRegressor, RFFNetClassifier
from src.models.rffnet.module import RFFLayer
from src.models.srff.srff import srf_run

import numpy as np

from src.models.rffnet.utils.datasets import (
    make_gregorova_se1,
    make_gregorova_se2,
    make_jordan_se1,
    make_jordan_se2,
    make_jordan_se3,
    make_piironen,
)


def run_model(X, y, X_val, y_val, X_test, y_test, model, is_classif, seed, args):
    n_features = X.shape[1]

    if is_classif:
        enc = LabelEncoder()
        enc.fit(y)
        n_classes = len(enc.classes_)

    if is_classif:
        if model == "rffnet":
            torch.manual_seed(seed)
            estimator = RFFNetClassifier(
                n_random_features=args.n_random_features,
                alpha=args.alpha,
                lr=args.learning_rate,
                n_restarts=0,
                max_iter=args.max_iter,
                batch_size=args.batch_size,
                n_iter_no_change=args.n_iter_no_change,
                random_state=seed,
                verbose=False,
            )
            estimator.fit(X, y)
            fit_time = estimator.fit_time
            relevances = estimator.relevances_

        if model == "fastfood":
            fastfood = Fastfood(
                sigma=n_features,
                n_components=args.n_random_features,
                tradeoff_mem_accuracy="mem",
                random_state=seed,
            )
            C = 1 / args.alpha if args.alpha > 0 else 0
            estimator = Pipeline(
                steps=[
                    ("feature_map", fastfood),
                    (
                        "logistic",
                        LogisticRegression(C=C, random_state=seed),
                    ),
                ]
            )
            t_start = time.time()
            estimator.fit(X, y)
            fit_time = time.time() - t_start
            relevances = None

        if model == "nystroem":
            nystroem = Nystroem(
                gamma=1 / n_features,
                n_components=args.n_random_features,
                random_state=seed,
            )
            C = 1 / args.alpha if args.alpha > 0 else 0
            estimator = Pipeline(
                steps=[
                    ("feature_map", nystroem),
                    ("logistic", LogisticRegression(C=C, random_state=seed)),
                ]
            )
            t_start = time.time()
            estimator.fit(X, y)
            fit_time = time.time() - t_start
            relevances = None

        if model == "eigenpro":
            estimator = EigenProClassifier(gamma=1 / n_features, random_state=seed)
            t_start = time.time()
            estimator.fit(X, y)
            fit_time = time.time() - t_start
            relevances = None

        if model == "logistic":
            C = 1 / args.alpha if args.alpha > 0 else 0

            estimator = LogisticRegression(C=C, random_state=seed)
            t_start = time.time()
            estimator.fit(X, y)
            fit_time = time.time() - t_start
            relevances = None

        if model == "nn":
            torch.manual_seed(seed)
            nn_seq = torch.nn.Sequential(
                nn.Linear(X.shape[1], 300),
                nn.ReLU(),
                nn.Linear(300, 20),
                nn.ReLU(),
                nn.Linear(20, 10),
                nn.ReLU(),
                nn.Linear(10, n_classes),
            )
            estimator = NeuralNetClassifier(
                module=nn_seq,
                max_epochs=args.max_iter,
                optimizer=torch.optim.Adam,
                criterion=torch.nn.CrossEntropyLoss,
                lr=args.learning_rate,
                batch_size=args.batch_size,
                callbacks=[
                    EarlyStopping(patience=args.n_iter_no_change),
                    Checkpoint(
                        f_params=None,
                        f_history=None,
                        f_optimizer=None,
                        f_pickle=None,
                        f_criterion=None,
                        load_best=True,
                    ),
                ],
                verbose=False,
            )
            t_start = time.time()
            estimator.fit(X, y)
            fit_time = time.time() - t_start
            relevances = None

        if model == "nn_rff":
            torch.manual_seed(seed)
            nn_rff = nn.Sequential(
                RFFLayer(n_features, args.n_random_features),
                nn.Linear(args.n_random_features, 20),
                nn.ReLU(),
                nn.Linear(20, n_classes),
            )

            estimator = NeuralNetClassifier(
                module=nn_rff,
                max_epochs=args.max_iter,
                optimizer=torch.optim.Adam,
                criterion=torch.nn.CrossEntropyLoss,
                lr=args.learning_rate,
                batch_size=args.batch_size,
                callbacks=[
                    EarlyStopping(patience=args.n_iter_no_change),
                    Checkpoint(
                        f_params=None,
                        f_history=None,
                        f_optimizer=None,
                        f_pickle=None,
                        f_criterion=None,
                        load_best=True,
                    ),
                ],
                verbose=False,
            )
            t_start = time.time()
            estimator.fit(X, y)
            fit_time = time.time() - t_start
            relevances = estimator.module[0].relevances.detach().numpy()
    else:
        if model == "rffnet":
            torch.manual_seed(seed)
            estimator = RFFNetRegressor(
                n_random_features=args.n_random_features,
                alpha=args.alpha,
                lr=args.learning_rate,
                n_restarts=0,
                max_iter=args.max_iter,
                batch_size=args.batch_size,
                n_iter_no_change=args.n_iter_no_change,
                random_state=seed,
                verbose=False,
            )
            estimator.fit(X, y)
            fit_time = estimator.fit_time
            relevances = estimator.relevances_

        if model == "kernel_ridge":
            estimator = KernelRidge(alpha=args.alpha, kernel="rbf", gamma=1 / n_features)
            estimator.fit(X, y)
            t_start = time.time()
            estimator.fit(X, y)
            fit_time = time.time() - t_start
            relevances = None

        if model == "fastfood":
            feature_map = Fastfood(
                sigma=n_features,
                n_components=args.n_random_features,
                tradeoff_mem_accuracy="mem",
                random_state=seed,
            )
            X = feature_map.fit_transform(X)
            X_test = feature_map.transform(X_test)

            estimator = Ridge(alpha=args.alpha, random_state=seed)
            t_start = time.time()
            estimator.fit(X, y)
            fit_time = time.time() - t_start
            relevances = None

        if model == "nystroem":
            feature_map = Nystroem(
                gamma=1 / n_features,
                n_components=args.n_random_features,
                random_state=seed,
            )
            X = feature_map.fit_transform(X)
            X_test = feature_map.transform(X_test)

            estimator = Ridge(alpha=args.alpha, random_state=seed)
            t_start = time.time()
            estimator.fit(X, y)
            fit_time = time.time() - t_start
            relevances = None

        if model == "eigenpro":
            estimator = EigenProRegressor(gamma=1 / n_features, random_state=seed)
            t_start = time.time()
            estimator.fit(X, y)
            fit_time = time.time() - t_start
            relevances = None

        if model == "gpr":
            torch.manual_seed(seed)
            estimator = ExactGPRegressor(
                ARDModule,
                optimizer=torch.optim.Adam,
                max_epochs=args.max_iter,
                lr=args.learning_rate,
                module__n_features=X.shape[1],
                verbose=False,
            )
            t_start = time.time()
            estimator.fit(X, y)
            fit_time = time.time() - t_start
            relevances = (
                1 / estimator.module_.covar_module.lengthscale.detach().numpy().flatten()
            )

        if model == "srff":
            torch.manual_seed(seed)
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y.reshape(-1, 1)).float()
            X_test = torch.from_numpy(X_test).float()
            y_test = torch.from_numpy(y_test.reshape(-1, 1)).float()
            X_val = torch.from_numpy(X_val).float()
            y_val = torch.from_numpy(y_val.reshape(-1, 1)).float()

            train_results, _, test_results = srf_run(X, y, X_val, y_val, X_test, y_test)
            fit_time = np.array(
                [train_results[i]["time"] / 1e3 for i in range(len(train_results))]
            )
            relevances = test_results["gamma"].detach().numpy()

        if model == "nn":
            torch.manual_seed(seed)
            nn_seq = torch.nn.Sequential(
                nn.Linear(X.shape[1], 300),
                nn.ReLU(),
                nn.Linear(300, 20),
                nn.ReLU(),
                nn.Linear(20, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
            )
            estimator = NeuralNetRegressor(
                module=nn_seq,
                max_epochs=args.max_iter,
                optimizer=torch.optim.Adam,
                criterion=torch.nn.MSELoss,
                lr=args.learning_rate,
                batch_size=args.batch_size,
                callbacks=[
                    EarlyStopping(patience=args.n_iter_no_change),
                    Checkpoint(
                        f_params=None,
                        f_history=None,
                        f_optimizer=None,
                        f_pickle=None,
                        f_criterion=None,
                        load_best=True,
                    ),
                ],
                verbose=False,
            )
            t_start = time.time()
            estimator.fit(X, y.reshape(-1, 1))
            fit_time = time.time() - t_start
            relevances = None

        if model == "nn_rff":
            torch.manual_seed(seed)
            nn_rff = nn.Sequential(
                RFFLayer(n_features, args.n_random_features),
                nn.Linear(args.n_random_features, 20),
                nn.ReLU(),
                nn.Linear(20, 1),
            )

            estimator = NeuralNetRegressor(
                module=nn_rff,
                optimizer=torch.optim.Adam,
                max_epochs=args.max_iter,
                criterion=nn.MSELoss,
                lr=args.learning_rate,
                batch_size=args.batch_size,
                callbacks=[
                    EarlyStopping(patience=args.n_iter_no_change),
                    Checkpoint(
                        f_params=None,
                        f_history=None,
                        f_optimizer=None,
                        f_pickle=None,
                        f_criterion=None,
                        load_best=True,
                    ),
                ],
                verbose=False,
            )
            t_start = time.time()
            estimator.fit(X, y.reshape(-1, 1))
            fit_time = time.time() - t_start
            relevances = estimator.module[0].relevances.detach().numpy()

    # Predict
    if is_classif:
        if model == "eigenpro":
            y_pred_raw = estimator._raw_predict(X_test)
            y_proba = np.exp(y_pred_raw) / sum(np.exp(y_pred_raw))
        else:
            y_proba = estimator.predict_proba(X_test)
        y_pred = estimator.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba[:, -1])
        metrics = (acc, roc_auc)

    else:
        if model == "srff":
            y_pred = test_results["preds"].numpy()
        else:
            y_pred = estimator.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        metrics = mse

    return y_pred, relevances, fit_time, metrics


def get_folder(folder_path, verbose=True):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        if verbose:
            print(f"-created directory {folder_path}")
    return folder_path


def get_generator(dataset):
    if dataset == "gse1":
        generator = make_gregorova_se1

    if dataset == "gse2":
        generator = make_gregorova_se2

    if dataset == "jse1":
        generator = make_jordan_se1

    if dataset == "jse2":
        generator = make_jordan_se2

    if dataset == "jse3":
        generator = make_jordan_se3

    if dataset == "piironen":
        generator = make_piironen

    if dataset == "moons":
        generator = make_moons

    if dataset == "circles":
        generator = make_circles

    if dataset == "blobs":
        generator = make_blobs

    if dataset == "hastie_10_2":
        generator = make_hastie_10_2

    if dataset == "classification":
        generator = make_classification

    return generator


def get_support(dataset):
    if dataset == "gse1":
        support = np.array([0, 2, 6, 7, 8])

    if dataset == "gse2":
        support = np.arange(10, 15)

    if dataset == "jse1":
        support = np.array([0])

    if dataset in ["jse2", "jse3"]:
        support = np.array([0, 1])

    if dataset == "piironen":
        support = np.arange(0, 8)

    return support


def get_xticks(dataset):
    if dataset == "gse1":
        x_ticks = np.array([1, 3, 7, 8, 9])
        x_labels = x_ticks
        x_lim = (0, 19)

    if dataset == "gse2":
        x_ticks = [11, 12, 13, 14, 15]
        x_labels = [11, "", "", "", 15]
        x_lim = (0, 30)

    if dataset == "jse1":
        x_ticks = [1]
        x_labels = [1]
        x_lim = (0, 11)

    if dataset in ["jse2", "jse3"]:
        x_ticks = [1, 2]
        x_labels = [1, 2]
        x_lim = (0, 11)

    return x_ticks, x_labels, x_lim
