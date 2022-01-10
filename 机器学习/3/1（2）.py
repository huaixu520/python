import random
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # 初始化w向量和截距b
    w = [0.0, 0.0]
    b = 0.0
    # learning_rate=0.1表示为学习率为0.1，step表示为训练次数
    learning_rate = 0.1
    step = 1000
    # 可分数据和相应的类标记
    samples = [[3, -2], [4, -3], [0, 1], [2, -1], [2, 1], [1, 2]]
    labels = [-1, -1, 1, -1, 1, 1]
    x1 = [0, 2, 1]
    y1 = [1, 1, 2]
    x2 = [3, 4, 2]
    y2 = [-2, -3, -1]
    w = np.array(w)
    for i in range(step):
        random_index = random.randint(0, len(samples) - 1)
        random_data = samples[random_index]
        # 如果y(wx + b) <= 0，则表明这个点为误分类点
        if random_data[1] * (w[0] * random_data[0] + w[1] * random_data[1] + b) <= 0:
            # 梯度下降算法更新w和b
            w[0] = w[0] + random_data[1] * random_data[0] * learning_rate
            w[1] = w[1] + random_data[1] * random_data[1] * learning_rate
            b = b + random_data[1] * learning_rate
            print("w: ", w)
            print("b: ", b)
    plt.scatter(x1, y1, label="y = 1")
    plt.scatter(x2, y2, label="y = -1")
    plt.legend()  # 图例
    plt.xlabel('X1')
    plt.ylabel('X2')
    label_x = np.linspace(0, 4, 20)
    label_y = -(b + w[0] * label_x) / w[1]
    plt.plot(label_x, label_y)
    plt.show()
