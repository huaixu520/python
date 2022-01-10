from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

store = pd.read_csv('Advertising.csv')

plt.figure(1)
sns1 = sns.regplot('TV', 'Sales', store)
plt.figure(2)
sns2 = sns.regplot('Radio', 'Sales', store)
plt.figure(3)
sns3 = sns.regplot('Newspaper', 'Sales', store)
plt.show()

modle = LinearRegression()
X_train, X_test, Y_train, Y_test = train_test_split(store.iloc[:, :3], store.Sales, train_size=0.8)
modle.fit(X_train, Y_train)
a = modle.coef_  # 回归系数
b = modle.intercept_  # 截距
print("拟合结果：截距", b, "回归系数：", a)
print(modle.predict(X_test))
print(Y_test)
