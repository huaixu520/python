import pandas as pd
import torch.nn as nn
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler  # 数据标准化模块
from sklearn.model_selection import train_test_split
from numpy import log1p
from sklearn.preprocessing import FunctionTransformer

# 读取数据
file = "预测.xlsx"
data0 = pd.read_excel(file)  # reading file
df = pd.DataFrame(data0)
df.dropna(axis=0, inplace=True)  # 删除带有空值的行
columns = df.columns

name = []

# 数据处理：
for i in columns:
    lists1 = data0.groupby([i])
    if 2 <= len(lists1) < 100:  # 判断长度在2-100之间的组
        name.append(i)

data = df[name]
data = pd.DataFrame(data)
columns1 = data.columns
for j in columns1:
    lists = data.groupby([j])
    for k in range(len(lists)):
        data[j].replace('{}'.format(list(lists)[k][0]), k, inplace=True)  # 替换


# 设置切分区域
listBins = [0, 30, 60]

# 设置切分后对应标签
listLabels = [0, 1]

# 利用pd.cut进行数据离散化切分
"""
pandas.cut(x,bins,right=True,labels=None,retbins=False,precision=3,include_lowest=False)
x:需要切分的数据
bins:切分区域
right : 是否包含右端点默认True，包含
labels:对应标签，用标记来代替返回的bins，若不在该序列中，则返回NaN
retbins:是否返回间距bins
precision:精度
include_lowest:是否包含左端点，默认False，不包含
"""
data['Survival_months'] = pd.cut(data['Survival_months'], bins=listBins, labels=listLabels, include_lowest=True)

Y = data['Survival_months']
Y = np.array(Y)

data.drop(columns='Survival_months', inplace=True)
#
data = pd.get_dummies(data)
# 标准化
X = np.array(data)
X = FunctionTransformer(log1p).fit_transform(X)
X = StandardScaler().fit_transform(X)

n_samples, n_features = data.shape  # 样本数量、特征数

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#
# 将数据转化为torch形式
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))
#
Y_train = Y_train.view(Y_train.shape[0], 1)  # 将标签多行1列
Y_test = Y_test.view(Y_test.shape[0], 1)


# 建立线性回归模型
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = Model(n_features)

num_epochs = 200
learning_rate = 0.03
criterion = nn.BCELoss()  # 损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 模型优化
sigmoid = nn.Sigmoid()  # 激活函数


for epoch in range(num_epochs):
    y_pred = model(X_train)
    loss = criterion(sigmoid(y_pred), Y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch + 1) % 20 == 0:
        print(f'epoch:{epoch + 1}, loss={loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()  # 四舍五入
    acc = y_predicted_cls.eq(Y_test).sum() / float(Y_test.shape[0])
    print(f'accuracy:{acc.item():.4f}')
