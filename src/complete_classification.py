import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X, y = datasets.make_classification(
    n_samples=300,
    n_features=5,
    n_classes=3,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.5, 0.3, 0.2],
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pipeline = make_pipeline(StandardScaler(), LogisticRegression())

pipeline.fit(X_train, y_train)

print(pipeline.score(X_test, y_test))
print(pipeline.score(X_train, y_train))
