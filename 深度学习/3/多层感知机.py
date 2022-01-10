from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD


'''Sequential： Keras 中的序贯模型，其特点是多个网络层线性堆叠
Dense：全连接神经元层
Dropout：神经元输入的断接率
Activation：神经元层的激励函数
SGD：优化器（optimizers）中的随机梯度下降法'''

crop = transforms.CenterCrop(30)
train_dir = r'./trainImages/'
test_dir = r'./testImages/'
x_train = []
y_train = []
x_test = []
y_test = []

#  训练集
for filename1 in os.listdir(train_dir):
    if filename1.endswith('jpg') or filename1.endswith('png'):
        filename1 = os.path.join(train_dir, filename1)
        y = int(filename1[14: 15])
        y_train.append(y)
        img = Image.open(filename1).convert('L')
        croped_img = crop(img)
        arr = np.array(croped_img)
        x_train.append(arr)
x_train = np.array(x_train)
X_train = x_train / 255  # 归一化数据集，将范围为 [0, 255] 的数据归一为 [0, 1]
X_train = X_train.reshape(120, -1)
Y_train = np.zeros((120, 10))
for i in range(len(y_train)):
    Y_train[i, y_train[i]] = 1

#  测试集
for filename2 in os.listdir(test_dir):
    if filename2.endswith('jpg') or filename2.endswith('png'):
        filename2 = os.path.join(test_dir, filename2)
        y = int(filename2[13: 14])
        y_test.append(y)
        img = Image.open(filename2).convert('L')
        croped_img = crop(img)
        arr = np.array(croped_img)
        x_test.append(arr)
x_test = np.array(x_test)
X_test = x_test / 255  # 归一化数据集，将范围为 [0, 255] 的数据归一为 [0, 1]
X_test = X_test.reshape(30, -1)
Y_test = np.zeros((30, 10))
for i in range(len(y_test)):
    Y_test[i, y_test[i]] = 1

'''
Sequential()：选用序贯模型进行训练及识别
input_shape=(30*30,)：输入为 900 维向量。
Dense(10)：输出 10 维向量
relu：隐藏层的激活函数
softmax：输出层的激活函数'''
batch_size = 128  # Sequential()：选用序贯模型进行训练及识别
epochs = 50  # 迭代次数
model = Sequential()
model.add(Dense(500, input_shape=(30 * 30,)))  # 每层 500 个神经元，输入层、隐藏层、输出层的连接为 Dense 全连接方式。
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
'''sgd：随机梯度下降法
lr=0.01：学习率
decay=1e-6：每次更新后的学习率衰减值
momentum=0.9：动量参数
nesterov=True：确定是否使用Nesterov动量
optimizer = sgd：优化器选为随机梯度下降法
loss = 'categorical_crossentropy'：损失函数选为多类的对数损失，适用于二值序列
metrics = ['accuracy']：指标列表，用于性能评估，一般设置为 metrics=['accuracy']
'''
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, Y_test))
result = pd.DataFrame(history.history)
print(result)
score = model.evaluate(X_test, Y_test,
                       batch_size=batch_size, verbose=1)
print(score)
