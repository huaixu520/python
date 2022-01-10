import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("Advertising.csv")

data = data.rename(columns={'Unnamed: 0': 'X0'})
data.iloc[:, 0:1] = 1
train, test = train_test_split(data, test_size=0.8)
X = np.array(train.iloc[:, 0:4])
Y = np.array(train.iloc[:, 4:5])

lr = 0.0001
# 二元线性模型: y_data = theta0 + theta1 * x_data[0] + theta2 * x_data[1] + theta3 * x_data[2]
theta0_init = 0
theta1_init = 0
theta2_init = 0
theta3_init = 0
# 最大迭代次数
ite = 10


# 梯度下降
def gradient_descent_runner(theta0, theta1, theta2, theta3, lr, ite, x_data, y_data):
    m = float(len(x_data))  # 样本容量
    for i in range(ite):
        theta0_grad = 0
        theta1_grad = 0
        theta2_grad = 0
        theta3_grad = 0
        for j in range(0, len(x_data)):
            theta0_grad += (1 / m) * ((theta0 + theta1 * x_data[j, 0] + theta2 * x_data[j, 1]) + theta3 * x_data[j, 2]
                                      - y_data[j])
            theta1_grad += (1 / m) * ((theta0 + theta1 * x_data[j, 0] + theta2 * x_data[j, 1]) + theta3 * x_data[j, 2]
                                      - y_data[j]) * x_data[j, 0]
            theta2_grad += (1 / m) * ((theta0 + theta1 * x_data[j, 0] + theta2 * x_data[j, 1]) + theta3 * x_data[j, 2]
                                      - y_data[j]) * x_data[j, 1]
            theta3_grad += (1 / m) * ((theta0 + theta1 * x_data[j, 0] + theta2 * x_data[j, 1]) + theta3 * x_data[j, 2]
                                      - y_data[j]) * x_data[j, 2]
        # 同步更新参数
        theta0 -= lr * theta0_grad
        theta1 -= lr * theta1_grad
        theta2 -= lr * theta2_grad
        theta3 -= lr * theta3_grad
    return theta0, theta1, theta2, theta3


if __name__ == '__main__':
    print('Starting theta0 = {0}, theta1 = {1}, theta2 = {2}, theta3 = {3}'.
          format(theta0_init, theta1_init, theta2_init, theta3_init))
    print('Gradient Descent Running...')
    theta0_end, theta1_end, theta2_end, theta3_end = gradient_descent_runner(theta0_init, theta1_init, theta2_init,
                                                                             theta3_init, lr, ite, X, Y)
    print('After {0} iterations theta0 = {1}, theta1 = {2}, theta2 = {3}, theta3 = {4}'.
          format(ite, theta0_end, theta1_end, theta2_end, theta3_end))
