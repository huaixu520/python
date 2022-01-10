from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
print(X_train.shape)
Y_train = np.zeros((120, 10))
for i in range(len(y_train)):
    Y_train[i, y_train[i]] = 1
print(Y_train.shape)

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
print(X_test.shape)
Y_test = np.zeros((120, 10))
for i in range(len(y_test)):
    Y_test[i, y_test[i]] = 1
print(Y_test.shape)

# transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
# data/表示下载数据集到的目录，transformation表示对数据集进行的相关处理
train_dataset = X_train
test_dataset = X_test
train_loader = Y_train
test_loader = Y_test
# 将数据加载为一个迭代器，读取其中一个批次
simple_data = next(iter(train_loader))


class Mnist_Net(nn.Module):
    def __init__(self):
        super(Mnist_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)  # 320是根据卷积计算而来4*4*20(4*4表示大小,20表示通道数)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x,p=0.1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Mnist_Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 优化函数


def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == "training":  # 判断当前是训练还是验证
        model.train()
    if phase == "validation":
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):  # 取出数据
        data, target = data, target
        data, target = Variable(data, volatile), Variable(target)
        if phase == 'training':
            optimizer.zero_grad()  # 重置梯度
        output = model(data)  # 得出预测结果
        loss = F.nll_loss(output, target)  # 计算损失值
        running_loss += F.nll_loss(output, target, size_average=False).item()  # 计算总的损失值
        preds = output.data.max(dim=1, keepdim=True)[1]  # 预测概率值转换为数字
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()
    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct / len(data_loader.dataset)
    print(
        f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss, accuracy


train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []
for epoch in range(1, 40):
    epoch_loss, epoch_accuracy = fit(epoch, model, train_loader, phase='training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch, model, test_loader, phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
from torchvision import models

transfer_model1 = models.resnet18(pretrained=True)
# 第一层卷积层改为1通道，因为mnist是(1,28,28)
transfer_model1.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# 输出的模型结构改为10维
dim_in = transfer_model1.fc.in_features
transfer_model1.fc = nn.Linear(dim_in, 10)
# 损失函数
criteon = nn.CrossEntropyLoss()
optimizer = optim.SGD(transfer_model.parameters(), lr=0.01)
transfer_model = transfer_model.cuda()
train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []
for epoch in range(10):
    transfer_model.train()
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        # 预测值logits
        x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        logits = transfer_model(x)

        loss = criteon(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = logits.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = 100 * running_correct / len(train_loader.dataset)
    train_losses.append(train_loss)
    train_accuracy.append(train_acc)
    print('epoch:{},train loss is{},train_acc is {}'.format(epoch, train_loss, train_acc))

    test_loss = 0.0
    test_acc_num = 0
    # 模型test
    model.eval()
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        logits = transfer_model(data)
        test_loss += criteon(logits, target).item()
        _, pred = torch.max(logits, 1)
        test_acc_num += pred.eq(target).float().sum().item()
    test_los = test_loss / len(test_loader.dataset)
    test_acc = test_acc_num / len(test_loader.dataset)
    val_losses.append(test_los)
    val_accuracy.append(test_acc)
    print("epoch:{} total loss:{},acc:{}".format(epoch, test_los, test_acc))
