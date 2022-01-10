1.

(1)β：对应u，即样本权重 w：对应coreectness，即每次训练集中分类正确样本权重之和比例和分类错误样本权重之和比例的比的平方根

(2)α：对应alpha，logw

(3)G(x)：对应prediction，即所有分类器结果和分类器权重的乘积的和，得到集成学习的预测结果

(4)基分类器实验原理：

此基分类器使用的单层决策树，其作用于数据集中的某一个特征，按照一个设定的阈值和分类方式将样本分为两类

为了缩小单层决策树的误差，这里使用求最优单层决策树算法，即求出一个单层决策树的最优的特征、最优的阈值、最优的分类方式，将

其分类误差最小化。通过三层循环，遍历所有特征，根据设定的步长遍历一定数量的阈值，遍历两种分类方式，对比所有的参数取值的误差结果

，将使误差结果最小的参数设为最优决策树的参数，即得到最优单层决策树。

2.

import cv2

import numpy as np

img1=cv2.imread("1.jpg")

img2=cv2.imread("2.jpg")

ORB_DESC=cv2.ORB_create()#创建ORB实例

kpoint1,descriptor1=ORB_DESC.detectAndCompute(img1,None)  #用ORB算法检测并计算给定图像的关键点

kpoint2,descriptor2=ORB_DESC.detectAndCompute(img2,None)

#绘制两张突破的关键点，关键点为白色且有大小和方向

keyPointImg1=cv2.drawKeypoints(img1,kpoint1,np.array([]),color=(255,255,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

keyPointImg2=cv2.drawKeypoints(img2,kpoint2,np.array([]),color=(255,255,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

matcher=cv2.BFMatcher(cv2.NORM_HAMMING)#实例化匹配器类，按海明距离计算每个关键点的描述子的相似度

matches=matcher.knnMatch(descriptor1,descriptor2,k=1)#按k近邻算法，参数用每张图像关键点的描述子来计算两张图像的最佳关键点匹配结果，k=1表示返回最佳匹配

img3=cv2.drawMatchesKnn(img1,kpoint1,img2,kpoint2,matches[:20],None)#将两张图象匹配的关键点连线，只绘制前20个最佳匹配关键点

cv2.namedWindow("1",0)

cv2.namedWindow("2",0)

cv2.namedWindow("Best_Match",0)

cv2.resizeWindow('1', 600, 800)

cv2.resizeWindow('2',  800, 600)

cv2.resizeWindow('Best_Match',  800, 600)

cv2.imshow("1",keyPointImg1)

cv2.imshow("2",keyPointImg2)

cv2.imshow("Best_Match",img3)

cv2.waitKey(0)

cv2.destroyAllWindows()

3.

import numpy as np

import pickle

from sklearn import metrics

def Find_k_min(arr,k):

    k_min_index=[]

    li=list(arr)

    for i in range(k):

        m=min(li)

        k_min_index.append(li.index(m))

        li.remove(m)

    return k_min_index

with open("data_batch_1",'rb')as fo:

    datadict=pickle.load(fo,encoding='bytes')

    X_train=datadict[b'data']

    Y_train=datadict[b'labels']

    Y_train=np.array(Y_train)

with open("test_batch",'rb')as fo:

    datadict=pickle.load(fo,encoding='bytes')

    X_test=datadict[b'data']

    Y_test=datadict[b'labels']

    Y_test=np.array(Y_test)



Y_predict=[]

for i in range(100):#因为测试数据过大，仅导入100个样本

    diff=np.tile(X_test[i],(10000,1))-X_train

    distances=np.linalg.norm(diff,axis=1)

    min_index=np.argmin(distances)

    predictY=Y_train[min_index]

    Y_predict.append(predictY)

Y_predict=np.array(Y_predict)

acc1 = metrics.accuracy_score(Y_test[:100],Y_predict)

print("最近邻算法的精度:",acc1)



k=3

Y_predict=[]

for i in range(100):#因为测试数据过大，仅导入100个样本

    diff=np.tile(X_test[i],(10000,1))-X_train

    distances=np.linalg.norm(diff,axis=1)

    k_min_index=Find_k_min(distances,k)

    predictY_li=[]

    for i in k_min_index:

        predictY_li.append(Y_train[i])

    predictY=max(predictY_li,key=predictY_li.count)

    Y_predict.append(predictY)

Y_predict=np.array(Y_predict)

acc2 = metrics.accuracy_score(Y_test[:100],Y_predict)

print(k,"近邻算法的精度:",acc2)



k=5

Y_predict=[]

for i in range(100):#因为测试数据过大，仅导入100个样本

    diff=np.tile(X_test[i],(10000,1))-X_train

    distances=np.linalg.norm(diff,axis=1)

    k_min_index=Find_k_min(distances,k)

    predictY_li=[]

    for i in k_min_index:

        predictY_li.append(Y_train[i])

    predictY=max(predictY_li,key=predictY_li.count)

    Y_predict.append(predictY)

Y_predict=np.array(Y_predict)

acc3 = metrics.accuracy_score(Y_test[:100],Y_predict)

print(k,"近邻算法的精度:",acc3)

cout：

最近邻算法的精度: 0.25

3 近邻算法的精度: 0.26

5 近邻算法的精度: 0.24

提高精度的方法：

降维方法（主成分分析、核化线性降维、剧本线性嵌入等）来对图像数据进行处理，也可以用卷积神经网络来提取图像特征，对每幅图像重新表示来提高分类精度。
