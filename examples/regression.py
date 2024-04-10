import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.rffnet.utils.datasets import make_gregorova_se1
from src.models.rffnet.estimators import RFFNetRegressor

torch.manual_seed(1)

X, y = make_gregorova_se1(n_samples=10000, random_state=1)

model = RFFNetRegressor(
    random_state=1,
    verbose=True,
)
model.fit(X, y)

plt.stem((np.abs(model.relevances_)))
plt.show()
