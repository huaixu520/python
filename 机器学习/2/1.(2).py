import numpy as np
import pandas as pd
from numpy.linalg import inv  # 矩阵的逆
from numpy import dot  # 矩阵相乘
from sklearn.model_selection import train_test_split

data = pd.read_csv("Advertising.csv")


def f(x1, x2, x3):
    y = beta[0, 0] + beta[1, 0] * x1 + beta[2, 0] * x2 + beta[3, 0] * x3
    return y


data = data.rename(columns={'Unnamed: 0': 'X0'})
data.iloc[:, 0:1] = 1
train, test = train_test_split(data, test_size=0.8)
X = np.array(train.iloc[:, 0:4])
Y = np.array(train.iloc[:, 4:5])
beta = dot(inv(dot(X.T, X)), dot(X.T, Y))
print("模型参数：")
print(beta)

test = np.array(test.iloc[:, 1:5])
for x, y, z, t in test:
    print("预测结果：", f(x, y, z), "，真实结果：", t)
