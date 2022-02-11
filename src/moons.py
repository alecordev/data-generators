import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_moons(200, noise=0.3, random_state=42)

fig, ax = plt.subplots(figsize=(6, 6))
plt.xlabel("X0", fontsize=20)
plt.ylabel("X1", fontsize=20)
plt.scatter(X[:, 0], X[:, 1], s=60, c=y)
plt.show()
