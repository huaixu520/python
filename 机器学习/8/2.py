import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# sklearn实现
def sklearn_my():
    df = pd.read_table('dataSetForKmeans.txt', header=None)
    X = np.array(df)
    plt.scatter(X[:, 0], X[:, 1])
    kmeans = KMeans(n_clusters=4, init='random')
    kmeans.fit(X)
    colors = ['b', 'g', 'r', 'y']
    markers = ['o', 's', 'd', '_']
    X1 = X[:, 0]
    X2 = X[:, 1]
    for i, l in enumerate(kmeans.labels_):
        plt.plot(X1[i], X2[i], color=colors[l], marker=markers[l], ls='None')
    plt.show()


# # 自己实现
def update_cluster_center(data, z, c):
    c = np.zeros(c.shape)
    for i, n in zip(z, np.arange(0, len(z))):
        c[i] = c[i] + data[n]
    for i in np.arange(0, len(c)):
        a = np.sum(z == i)
        c[i] = c[i] / a
    return c


def my_kmeans(data, k, o=2, m=50):
    z = np.zeros(len(data))
    C = np.random.random((k, len(data[0])))  # C: 聚类中心矩阵
    for i in np.arange(0, m):
        n = len(data)
        k = len(C)
        d = np.zeros((n, k))
        for i in np.arange(0, n):
            for j in np.arange(0, k):
                d[i][j] = np.dot((data[i] - C[j]), (data[i] - C[j]))
        z = np.argmin(d, axis=1)
        C1 = update_cluster_center(data, z, C)
        n = len(C)
        k = len(C1)
        d = np.zeros((n, k))
        for i in np.arange(0, n):
            for j in np.arange(0, k):
                d[i][j] = np.dot((data[i] - C[j]), (data[i] - C[j]))
        dis = np.sqrt(np.sum(d))
        if dis < o:
            break
        C = C1
    return z


if __name__ == "__main__":
    data = np.loadtxt("dataSetForKmeans.txt")
    sklearn_my()
    z = my_kmeans(data, 3)
    plt.scatter(data[:, 0], data[:, 1], c=z)
    plt.show()
