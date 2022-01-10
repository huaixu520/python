import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

df = pd.read_table('dataSetForKmeans.txt', header=None)
X = np.array(df)
y = SpectralClustering().fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
