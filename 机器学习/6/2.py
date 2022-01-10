from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import time

start = time.time()
digits = load_digits()
train_size = 1500
train_x, train_y = digits.data[:train_size], digits.target[:train_size]
test_x, test_y = digits.data[train_size:], digits.target[train_size:]
tree = DecisionTreeClassifier(criterion='entropy')
# n_estimators基础学习器个数
bagging = BaggingClassifier(base_estimator=tree, n_estimators=50, max_samples=1.0, max_features=10)
bagging.fit(train_x, train_y)
end = time.time()
# for i in range(len(test_y)):
#     print("预测值：", bagging.predict(test_x)[i], " 真实值：", test_y[i])
print("模型精度:", bagging.score(test_x, test_y))
print('Total time: %.2f' % (end - start))
