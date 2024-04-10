import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_classification

from src.models.rffnet.estimators import RFFNetClassifier

torch.manual_seed(1)

X, y = make_classification(
    n_samples=10_000, n_classes=4, n_features=20, n_informative=5, random_state=42
)

model = RFFNetClassifier(
    n_random_features=500,
    lr=1e-3,
    max_iter=200,
    n_iter_no_change=30,
    random_state=1,
    verbose=True,
)
model.fit(X, y)

plt.stem(np.abs(model.relevances_))
plt.show()
