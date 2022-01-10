import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 读取数据
file = "预测.xlsx"
data1 = pd.read_excel(file)  # reading file
df = pd.DataFrame(data1)
df.dropna(axis=0, inplace=True)  # 删除带有空值的行
columns = df.columns

name = []

# 数据处理：
for i in columns:
    lists1 = data1.groupby([i])
    if 2 <= len(lists1) < 100:  # 判断长度在2-100之间的组
        name.append(i)

data = df[name]
data = pd.DataFrame(data)
columns1 = data.columns
for j in columns1:
    lists = data.groupby([j])
    for k in range(len(lists)):
        data[j].replace('{}'.format(list(lists)[k][0]), k, inplace=True)  # 替换

print(data.shape)
# 数据划分：
train_data = data.sample(frac=0.8, random_state=None, axis=0)  # 训练集

list0 = train_data.index.unique().tolist()
text_data = data.drop(index=data[data.index.isin(list0)].index.tolist())
print(train_data.shape)
print(text_data.shape)
train_data.to_excel('traindata.xlsx', index=0)
text_data.to_excel('textdata.xlsx', index=0)

file1 = "traindata.xlsx"
file2 = "textdata.xlsx"

data1 = pd.read_excel(file1)  # reading file
data2 = pd.read_excel(file2)  # reading file

data11 = pd.DataFrame(data1)
data22 = pd.DataFrame(data2)

y_train = data11['Survival_months']
data11.drop(columns='Survival_months', inplace=True)

y_test = data22['Survival_months']
data22.drop(columns='Survival_months', inplace=True)

x_train = np.array(data11)
y_train = np.array(y_train)
x_test = np.array(data22)
y_test = np.array(y_test)

x_train = x_train.reshape(train_data.shape[0], train_data.shape[1] - 1)
y_train = y_train.reshape(train_data.shape[0], 1)
x_test = x_test.reshape(text_data.shape[0], text_data.shape[1] - 1)
y_test = y_test.reshape(text_data.shape[0], 1)

# 模型预测
model = XGBClassifier()  # 载入模型
model.fit(x_train, y_train)  # 训练模型
y_pred = model.predict(x_test)  # 模型预测


# 平均绝对误差(MAE）
def MAE(y, y_pre):
    return abs(np.mean(y - y_pre))


# 均方误差（MSE）
def MSE(y, y_pre):
    return np.mean((y - y_pre) ** 2)


# R²评价指标
def R2(y, y_pre):
    return r2_score(y, y_pre)


print("MAE: ", MAE(y_test, y_pred))
print("MSE: ", MSE(y_test, y_pred))
print("R²: ", R2(y_test, y_pred))

plt.figure()
plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
plt.plot(range(len(y_pred)), y_test, 'r', label="test")
plt.legend(loc="upper right")  # 显示图中的标签
plt.show()
