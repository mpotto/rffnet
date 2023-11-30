import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.models.rffnet.utils.datasets import make_correlated_data
from src.models.rffnet.estimators import RFFNetRegressor
from src.models.rffnet.selection import SelectTopK

from src.models.rffnet.estimators import RFFNetEstimator
from src.models.rffnet.initialization import Constant
from src.models.rffnet.solvers import PALM
from src.models.rffnet.penalties import L1, L2

torch.manual_seed(1)

X, y, w = make_correlated_data(
    n_samples=5_000,
    n_features=5,
    rho=0.5,
    w_true=np.array([1, 0, 1, 0, 0]),
    random_state=0,
)
X, X_test, y, y_test = train_test_split(X, y, train_size=0.9, random_state=4)

datafit = torch.nn.MSELoss()
init = Constant()
solver = PALM(
    torch.optim.Adam,
    batch_size=128,
    lr=1e-3,
    max_iter=100,
    early_stopping=True,
    validation_fraction=2_000,
    n_iter_no_change=10,
    verbose=True,
    random_state=0,
)

model1 = RFFNetEstimator(
    n_random_features=300,
    datafit=datafit,
    initializer=init,
    solver=solver,
    penalty=(L2(1e-4), L1(2)),
)

model2 = RFFNetRegressor(
    n_random_features=50,
    alpha=1e-1,
    lr=1e-3,
    n_restarts=0,
    warm_restart=True,
    max_iter=100,
    max_init_iter=20,
    batch_size=100,
    n_iter_no_change=20,
    random_state=1,
    verbose=True,
)
model1.fit(X, y)
model2.fit(X, y)

topk = SelectTopK(model2)

results = topk.path(X_test, y_test, mean_squared_error, ks=list(range(5)))

for i in range(len(results[0])):
    print(w)
    plt.stem(np.abs(results[1][i].relevances_))
    print(results[0][i], results[2][i])
    plt.show()

print(np.abs(model1.relevances_))
print(mean_squared_error(model1.predict(X_test), y_test))
plt.stem(np.abs(model1.relevances_))
plt.show()
