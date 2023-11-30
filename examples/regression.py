import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.rffnet.utils.datasets import make_jordan_se1
from src.models.rffnet.estimators import RFFNetRegressor

torch.manual_seed(1)

X, y = make_jordan_se1(n_samples=500, n_features=1000, noise_level=1, random_state=0)

model = RFFNetRegressor(
    n_random_features=50,
    alpha=2,
    lr=1e-3,
    n_restarts=0,
    # sampler=sampler,
    warm_restart=False,
    max_iter=200,
    max_init_iter=10,
    batch_size=32,
    n_iter_no_change=None,
    random_state=1,
    verbose=True,
)
model.fit(X, y)

print(model.relevances_)

plt.stem(np.abs(model.relevances_))
plt.show()
