import torch as pt
import torchvision as ptv
import numpy as np

train_set = ptv.datasets.MNIST("./train", train=True, transform=ptv.transforms.ToTensor(), download=False)
test_set = ptv.datasets.MNIST("./test", train=False, transform=ptv.transforms.ToTensor(), download=False)
print(train_set)
print(test_set)
train_dataset = pt.utils.data.DataLoader(train_set, batch_size=100)
test_dataset = pt.utils.data.DataLoader(test_set, batch_size=100)


class MLP(pt.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = pt.nn.Linear(784, 512)
        self.fc2 = pt.nn.Linear(512, 128)
        self.fc3 = pt.nn.Linear(128, 10)

    def forward(self, din):
        din = din.view(-1, 28 * 28)
        dout = pt.nn.functional.relu(self.fc1(din))
        dout = pt.nn.functional.relu(self.fc2(dout))
        return pt.nn.functional.softmax(self.fc3(dout))


model = MLP()
print(model)

optimizer = pt.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
lossfunc = pt.nn.CrossEntropyLoss()


def AccuarcyCompute(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    # print(pred.shape, label.shape)
    test_np = (np.argmax(pred, 1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)


for x in range(4):
    for i, data in enumerate(train_dataset):
        optimizer.zero_grad()
        (inputs, labels) = data
        inputs = pt.autograd.Variable(inputs)
        labels = pt.autograd.Variable(labels)
        outputs = model(inputs)
        loss = lossfunc(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(i, ":", AccuarcyCompute(outputs, labels))

test_save_net = MLP()
test_save_net.load_state_dict(pt.load("./mlp_params.pt"))
test_save_model = pt.load("./mlp_model.pt")
accuarcy_list = []
for i, (inputs, labels) in enumerate(test_dataset):
    inputs = pt.autograd.Variable(inputs)
    labels = pt.autograd.Variable(labels)
    outputs = model(inputs)
    accuarcy_list.append(AccuarcyCompute(outputs, labels))
print(sum(accuarcy_list) / len(accuarcy_list))
# （1）数据集中样本数量是多少？每个样本的特征是多少？
# 训练集60000，测试集10000，每个样本特征为28*28=784
# （2）所构造的感知机有多少隐藏层？每个隐藏层有多少个神经元？
# 隐藏层两个，512个和128个神经元
# （3）测试所训练模型的精度。
# 0.902700001001358