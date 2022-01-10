import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from numpy import log1p
from sklearn.preprocessing import FunctionTransformer
from torch import optim

df = pd.read_excel('预测.xlsx')
f = lambda s: 1 if s["Survival_months"] >= 21 else 0
df["Survival_months"] = df.apply(f, axis=1)
# 删除明显意义不符合的和方差接近0的列
df.drop(['Patient ID', 'Age recode with <1 year olds', 'Race/ethnicity', 'Laterality', 'Behavior code ICD-O-3',
         'Type of Reporting Source', 'CS site-specific factor 25 (2004+ varying by schema)',
         'SEER historic stage A (1973-2015)', 'SEER summary stage 2000 (2001-2003)', 'Year of diagnosis',
         'Month of diagnosis'], axis=1, inplace=True)
# print(df)
# print(df.shape)

# 离散化+独热编码
df.dropna(axis=0, inplace=True)  # 删除带有空值的行
Y = np.array(df["Survival_months"])
df.drop("Survival_months", axis=1, inplace=True)
df = pd.get_dummies(df)
# print(df)

# 标准化
X = np.array(df)
X = FunctionTransformer(log1p).fit_transform(X)
X = StandardScaler().fit_transform(X)
X_train = X[:int(X.shape[0] * 0.8), :]
X_test = X[int(X.shape[0] * 0.8):, :]
Y_train = Y.reshape(37264, 1)[:int(Y.shape[0] * 0.8), :]
Y_test = Y.reshape(37264, 1)[int(Y.shape[0] * 0.8):, :]

X_train = torch.tensor(X_train).to(torch.float32)
X_test = torch.tensor(X_test).to(torch.float32)
Y_train = torch.tensor(Y_train).to(torch.float32)
Y_test = torch.tensor(Y_test).to(torch.float32)

# 建立线性回归模型
D_in, D_out = 123, 1
linear = nn.Linear(D_in, D_out, bias=True)
# output = linear(X_train)
# print(X_train.shape,linear.weight.shape,linear.bias.shape,output.shape)

# 激活函数
sigmoid = nn.Sigmoid()
# scores = sigmoid(output)

# 损失函数
loss = nn.BCELoss()
# loss(scores,Y_train)

# 模型优化
optimizer = optim.SGD(linear.parameters(), lr=0.03)
batch_size = 10
iters = 10
for _ in range(iters):
    for i in range(int(len(X_train) / batch_size)):
        input = X_train[i * batch_size:(i + 1) * batch_size]
        target = Y_train[i * batch_size:(i + 1) * batch_size]
        optimizer.zero_grad()
        output = linear(input)
        print(sigmoid(output))
        print(target)
        l = loss(sigmoid(output), target)
        l.backward()
        optimizer.step()

output = linear(X_test)
scores = sigmoid(output)
print('优化后损失函数：', loss(scores, Y_test))
print('分类结果：', scores)
print('测试结果', Y_test)
