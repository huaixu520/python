import numpy as np
from sklearn.linear_model import Perceptron
from matplotlib import pyplot as plt

samples = np.array([[3, -2], [4, -3], [0, 1], [2, -1], [2, 1], [1, 2]])
labels = np.array([-1, -1, 1, -1, 1, 1])
x1 = [0, 2, 1]
y1 = [1, 1, 2]
x2 = [3, 4, 2]
y2 = [-2, -3, -1]
per = Perceptron(fit_intercept=True, max_iter=30, shuffle=False)
per.fit(samples, labels)
acc = per.score(samples, labels)
print(acc)

# 画出正例和反例的散点图
plt.scatter(x1, y1, label="y = 1")
plt.scatter(x2, y2, label="y = -1")
# 画出超平面（在本例中即是一条直线）
line_x = np.arange(0, 6)
line_y = line_x * (-per.coef_[0][0] / per.coef_[0][1]) - per.intercept_ / per.coef_[0][1]
plt.plot(line_x, line_y)
plt.show()
