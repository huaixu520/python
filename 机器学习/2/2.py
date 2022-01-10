import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('Wage.csv')
data.replace("1. Male", 1, inplace=True)

data.replace("1. Never Married", 1, inplace=True)
data.replace("2. Married", 2, inplace=True)
data.replace("3. Widowed", 3, inplace=True)
data.replace("4. Divorced", 4, inplace=True)
data.replace("5. Separated", 5, inplace=True)

data.replace("1. White", 1, inplace=True)
data.replace("2. Black", 2, inplace=True)
data.replace("3. Asian", 3, inplace=True)
data.replace("4. Other", 4, inplace=True)

data.replace("1. < HS Grad", 1, inplace=True)
data.replace("2. HS Grad", 2, inplace=True)
data.replace("3. Some College", 3, inplace=True)
data.replace("4. College Grad", 4, inplace=True)
data.replace("5. Advanced Degree", 5, inplace=True)

data.replace("2. Middle Atlantic", 2, inplace=True)

data.replace("1. Industrial", 1, inplace=True)
data.replace("2. Information", 2, inplace=True)

data.replace("1. <=Good", 1, inplace=True)
data.replace("2. >=Very Good", 2, inplace=True)

data.replace("1. Yes", 1, inplace=True)
data.replace("2. No", 2, inplace=True)

#
plt.figure(1)
sns1 = sns.regplot('year', 'wage', data)
plt.figure(2)
sns2 = sns.regplot('age', 'wage', data)
plt.figure(3)
sns3 = sns.regplot('sex', 'wage', data)
plt.figure(4)
sns3 = sns.regplot('maritl', 'wage', data)
plt.figure(5)
sns3 = sns.regplot('race', 'wage', data)
plt.figure(6)
sns3 = sns.regplot('education', 'wage', data)
plt.figure(7)
sns3 = sns.regplot('region', 'wage', data)
plt.figure(8)
sns3 = sns.regplot('jobclass', 'wage', data)
plt.figure(9)
sns3 = sns.regplot('health', 'wage', data)
plt.figure(10)
sns3 = sns.regplot('health_ins', 'wage', data)
#plt.show()

df = pd.DataFrame(data)
df = df.drop('Unnamed: 0', axis=1)  # 删除第一列
#df = df.drop('logwage', axis=1)  # 删除倒数第二列
df = df.drop('wage', axis=1)  # 删除最后一列
X_train, X_test, Y_train, Y_test = train_test_split(df, data.wage, train_size=0.8)
model1 = LinearRegression()  # 构建模型
model1.fit(X_train, Y_train)  # 训练模型
score = model1.score(X_train, Y_train)  # 评估模型
print("模型评估值：", score)
a = model1.coef_  # 回归系数
b = model1.intercept_  # 截距
print("拟合结果：截距", b, "回归系数：", a)
print("____________________________________________________")

for i, col in enumerate(df.columns):
    X_train, X_test, Y_train, Y_test = train_test_split(df.loc[:, col].values.reshape(-1, 1), data.wage, train_size=0.8)
    # 每一次循环，都取出datafram中的一列数据，是一维Series数据格式，但是线性回归模型要求传入的是一个二维数据，因此利用reshape修改其形状
    model = LinearRegression()  # 构建模型
    model.fit(X_train, Y_train)  # 训练模型
    score = model.score(X_train, Y_train)  # 评估模型
    print(col)
    print("模型评估值：", score)
    a = model.coef_  # 回归系数
    b = model.intercept_  # 截距
    print("拟合结果：截距", b, "回归系数：", a)

"""logwage、education、jobclass对工资影响较大
sex、region对工资影响较小"""