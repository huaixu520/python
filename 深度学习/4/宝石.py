import math

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

# %%
batch_size = 8


# 训练数据108张 测试数据27张 总计135张
def load_train_data():
    path = r"./archive_train"
    trainset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    # 一般输入深度网络的特征图长宽是相等的，就不能采取等比例缩放的方式了，需要同时指定长宽
                                                    transforms.Resize([43, 43]),
                                                    transforms.CenterCrop(43),
                                                    # 归一化
                                                    transforms.ToTensor(),
                                                    # 输入的每个channel做 ( [0, 1] - mean（0.5) )/ std（0.5）= [-1, 1] 的运算，所以这一句的实际结果是将[0，1]的张量归一化到[-1, 1]上
                                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                                ]))
    # PyTorch 数据加载的核心，负责加载数据，同时支持 Map-style 和 Iterable-style Dataset，
    # 默认情况下遍历dataloader其实就是输出一个batch内的图像和对应的label
    trainload = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainload


def load_test_data():
    path = r"./archive_test"
    testset = torchvision.datasets.ImageFolder(path,
                                               transform=transforms.Compose([
                                                   transforms.Resize([43, 43]),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                               ]))
    testload = DataLoader(testset, batch_size=batch_size, shuffle=True)
    return testload


# %%


# 加载训练集
trainloader = load_train_data()

# 加载测试集
testloader = load_test_data()
# 25分类
classes = ('Alexandrite', 'Almandine', 'Benitoite', 'Beryl Golden', 'Carnelian',
           'Cats Eye',  'Danburite', 'Diamond', 'Emerald', 'Fluorite',
           'Garnet Red', 'Hessonite', 'Iolite', 'Jade', 'Kunzite',
           'Labradorite', 'Malachite', 'Onyx Black', 'Pearl', 'Quartz Beer',
           'Rhodochrosite', 'Sapphire Bule', 'Tanzanite', 'Variscite', 'Zircon')
# %%




# 显示图像的函数

# 显示训练图片
def imshow(img):
    img = img / 2 + 0.5  # 不正常
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 得到一些随机的训练图像
dataiter = iter(trainloader)

images, labels = dataiter.next()
print(labels)
# 显示图片
imshow(torchvision.utils.make_grid(images))
# 训练打印标签
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

# %%




# 定义BasicBlock


# 常见的3x3卷积
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# 这是残差网络中的basicblock，实现的功能如下方解释：
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):  # inplanes代表输入通道数，planes代表输出通道数。
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4  # 输出通道数的倍乘

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):  # layers=参数列表 block选择不同的类
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))  # 每个blocks的第一个residual结构保存在layers列表中。
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))  # 该部分是将每个blocks的剩下residual 结构保存在layers列表中，这样就完成了一个blocks的构造。

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 将输出结果展成一行
        x = self.fc(x)

        return x


net= ResNet(BasicBlock, [1, 1, 1, 1], 25)

# %%



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


# %%

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # print(outputs)
        # print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 8 == 7:  # print every 8 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 8))
            running_loss = 0.0

print('Finished Training')

# %%

PATH = './gem_net.pth'  # 保存路径
torch.save(net.state_dict(), PATH)

# %%

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(8)))

# %%

net = ResNet(BasicBlock, [1, 1, 1, 1], 25)
net.load_state_dict(torch.load(PATH))

# %%

outputs = net(images)

# %%

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(8)))

# %%

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 163 test images: %d %%' % (
        100 * correct / total))

# %%

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("{:5s} is: {:.1f} %".format(classname,
                                                         accuracy))
