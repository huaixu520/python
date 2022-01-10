import numpy as np
import matplotlib.pyplot as plt

# 1.得到特征向量
# 2.将特征矩阵每行零均值化
def load_and_center_dataset(filename):
    data = np.load(filename).reshape((2000, 784))  # 加载数据并将其重塑为正确的形状
    return data - data.mean(axis=1).repeat(784).reshape((2000, 784))  # 将数据集中；这是dimension_reduction所必需的


# 3.求矩阵XX^T的特征值和特征向量，并对特征值从大到小排序，如果特征值和特征向量出现复数，请用numpy的real函数提取复数的实部就可以了，
# 比如调用的格式为：np.real(........)。
def dimension_reduction(dataset, k=784):
    # 协方差矩阵
    covariance_matrix = np.dot(dataset.T, dataset) / (dataset.shape[1] - 1)
    # 求特征值w和特征向量v
    w, v = np.linalg.eig(np.mat(covariance_matrix))
    # 去掉虚部
    w = np.real(w)
    v = np.real(v)
    # 前k个特征值的索引
    w_index = np.argsort(-w)
    # 取前k个特征值
    w2 = w[0:k]
    a = w2.sum() / w.sum()
    print("前", k, "个特征值此所占整个特征值的比重：", a)
    # 前k个特征值对应的特征向量
    Vects = v[:, w_index[0:k]]
    new_data = load_and_center_dataset(filename)
    # 还原维度
    reconData = np.array(np.real(new_data * Vects * Vects.T))
    # 绘图
    X = data
    Y = reconData
    d = X[[1]]
    e = Y[[1]]
    display_image(d, e)


# 4.取前20，50，100个特征向量作为投影矩阵，将数据通过投影矩阵得到新表示的样本，对原来的样本和新样本进行绘图来比较它们的差异性。并给出前20，
# 50，100个特征值此所占整个特征值的比重分别是多少？
def display_image(orig, proj):
    # 生成并显示相关绘图
    fig, axs = plt.subplots(ncols=2)
    a1 = axs[0].imshow(orig.reshape((28, 28)), aspect='equal', cmap='gray')
    axs[0].set_title("Original")
    a2 = axs[1].imshow(proj.reshape((28, 28)), aspect='equal', cmap='gray')
    axs[1].set_title("Projection")

    fig.colorbar(a1, ax=axs[0])
    fig.colorbar(a2, ax=axs[1])
    plt.show()


if __name__ == '__main__':
    filename = 'mnist.npy'
    data = np.load(filename).reshape(2000, 784)
    # 降成20维
    dimension_reduction(data, 20)
    dimension_reduction(data, 50)
    dimension_reduction(data, 100)
