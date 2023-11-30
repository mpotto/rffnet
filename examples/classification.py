import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_classification

from src.models.rffnet.estimators import RFFNetClassifier
from src.models.rffnet.initialization import Restarter
from src.models.rffnet.solvers import PALM
from src.models.rffnet.penalties import L2, Null

torch.manual_seed(1)

X, y = make_classification(
    n_samples=10_000, n_classes=4, n_features=20, n_informative=5, random_state=42
)

init = Restarter(max_iter=10, warm_restart=False)
solver = PALM(early_stopping=True, verbose=True, lr=1e-3, batch_size=32)
penalty = (L2(1e-4), Null())

model = RFFNetClassifier(
    n_random_features=100,
    alpha=1,
    lr=1e-3,
    batch_size=32,
    n_restarts=1,
    warm_restart=True,
    max_init_iter=10,
    n_iter_no_change=5,
    random_state=1,
    verbose=True,
)
model.fit(X, y)

print(model.predict(X[:10]), y[:10])
print(model.predict_proba(X[:10]))

plt.stem(np.abs(model.relevances_))
plt.show()
