import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_classification(
    n_samples=300,
    n_features=5,
    n_classes=3,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.5, 0.3, 0.2],
    random_state=42,
)

fig, ax = plt.subplots(figsize=(9, 6))
plt.xlabel("X0", fontsize=20)
plt.ylabel("X1", fontsize=20)
plt.scatter(X[:, 0], X[:, 1], s=50, c=y)
plt.show()
