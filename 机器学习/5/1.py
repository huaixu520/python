import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = open('ex5data.txt', 'r')
data = file.read()
data = data.split('\n')
data = data[:len(data) - 1]
data = [whole.split(',') for whole in data]
data = [[float(subpart) for subpart in part] for part in data]
data = np.array(data)
# Adding a column of ones in front
data = np.hstack((np.ones((data.shape[0], 1)), data))
# differentiating X and Y from mixed data
X = data[:, 0:3]
y = data[:, 3]
plt.scatter(X[y == 0, [1]], X[y == 0, [2]], color='yellow', marker='o', label='Not admitted')
plt.scatter(X[y == 1, [1]], X[y == 1, [2]], color='black', marker='+', label='admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(loc='best')
plt.show()


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def costFunction(initial_theta, X, y):
    cost = sum(-1 * (y * (np.log(sigmoid(np.dot(X, initial_theta)))) + (1 - y) *
                     (np.log(1 - (sigmoid(np.dot(X, initial_theta)))))))
    return cost


initial_theta = np.zeros(X.shape[1])
# print(initial_theta.shape)
cost = costFunction(initial_theta, X, y)
print(cost)
y = y.reshape(1, len(X))


def gradientFunction(X, y):
    learning_rate = 0.001
    step = 1500
    initial_theta = [0, 0, 0]
    initial = [0, 0, 0]
    initial_theta = np.array(initial_theta)
    for i in range(step):
        initial = initial_theta - learning_rate * (
            sum(np.dot((sigmoid(np.dot(X, initial_theta).reshape(-1, 100)) - y)
                       , (X.reshape(100, 3)))))
        initial_theta = initial

    return initial_theta


theta = gradientFunction(X, y)

# theta, J_history = gradientFunction(X, initial_theta)
print(theta)
a = sigmoid(np.dot(np.array([[1, 45, 85]]), theta))
print(np.dot(np.array([[1, 45, 85]]), theta))
# theta
print(a)

data1 = pd.read_csv('Iris.csv')
data1.replace("Iris-setosa", 0, inplace=True)
data1.replace("Iris-versicolor", 1, inplace=True)
data1.replace("Iris-virginica", 2, inplace=True)
a = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
data1 = data1[a]
data1 = np.array(data1)
X = data1[:, 0:4]
y = data1[:, 4]
plt.scatter(X[y == 0, [0]], X[y == 0, [1]], color='yellow', marker='o', label='Iris-setosa')
plt.scatter(X[y == 1, [0]], X[y == 1, [1]], color='black', marker='+', label='Iris-versicolor')
plt.scatter(X[y == 2, [0]], X[y == 2, [1]], color='red', marker='*', label='Iris-virginica')
# plt.xlabel('Exam 1 score')
# plt.ylabel('Exam 2 score')
# plt.legend(loc='best')
# plt.show()

plt.scatter(X[y == 0, [2]], X[y == 0, [3]], color='yellow', marker='o', label='Iris-setosa')
plt.scatter(X[y == 1, [2]], X[y == 1, [3]], color='black', marker='+', label='Iris-versicolor')
plt.scatter(X[y == 2, [2]], X[y == 2, [3]], color='red', marker='*', label='Iris-virginica')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(loc='best')
plt.show()


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def costFunction(initial_theta, X, y):
    cost = sum(-1 * (y * (np.log(sigmoid(np.dot(X, initial_theta)))) + (1 - y) *
                     (np.log(1 - (sigmoid(np.dot(X, initial_theta)))))))
    return cost


initial_theta = np.zeros(X.shape[1])
# print(initial_theta.shape)
cost = costFunction(initial_theta, X, y)
print(cost)
y = y.reshape(1, len(X))


def gradientFunction(X, y):
    learning_rate = 0.001
    step = 1500
    initial_theta = [0, 0, 0, 0]
    initial = [0, 0, 0, 0]
    initial_theta = np.array(initial_theta)
    for i in range(step):
        initial = initial_theta - learning_rate * (
            sum(np.dot((sigmoid(np.dot(X, initial_theta).reshape(-1, 150)) - y)
                       , (X.reshape(150, 4)))))
        initial_theta = initial

    return initial_theta


theta = gradientFunction(X, y)

# theta, J_history = gradientFunction(X, initial_theta)
print(theta)
a = sigmoid(np.dot(np.array([[4.8, 3.3, 1.7, 0.5]]), theta))
print(np.dot(np.array([[4.8, 3.3, 1.7, 0.5]]), theta))
# theta
print(a)
