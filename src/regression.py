import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(
    n_samples=200, n_features=5, n_informative=2, random_state=42
)

# Create Pandas Dataframe and processes correlation
# You could also use numpy corrcoef method for same

df = pd.DataFrame(X)
df.columns = ["ftre1", "ftre2", "ftre3", "ftre4", "ftre5"]
df["target"] = y

# Determine correlations

corr = df.corr()

# Draw the correlation heatmap
f, ax = plt.subplots(figsize=(9, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, annot=True, mask=mask, cmap=cmap)
plt.show()
