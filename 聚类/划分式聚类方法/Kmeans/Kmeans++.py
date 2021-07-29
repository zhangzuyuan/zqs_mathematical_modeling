# K-means和K-means++主要区别在于，K-means++算法选择初始类中心时，
# 尽可能选择相距较远的类中心，而K-means仅仅是随机初始化类中心。

import numpy as np
from numpy import random
import time
import matplotlib.pyplot as plt
from numpy.lib.shape_base import dsplit

MAX=100000.0
MININDEX=-1;

# 加载数据
def loadDataSet(fileName):
    data = np.loadtxt(fileName,delimiter='\t')
    return data

# 欧氏距离计算
def distEclud(x,y):
    return np.sqrt(np.sum((x-y)**2))  # 计算欧氏距离


#对一个样本找到与该样本距离最近的聚类中心
def nearest(point, cluster_centers):
    min_dist = MAX
    m = np.shape(cluster_centers)[0]  # 当前已经初始化的聚类中心的个数
    for i in range(m):
        # 计算point与每个聚类中心之间的距离
        d = distEclud(point, cluster_centers[i, ])
        # 选择最短距离
        if min_dist > d:
            min_dist = d
    return min_dist

# 为给定数据集构建一个包含K个随机质心的集合
#从数据的那几个点中随便挑选几个作为质心
#选择尽可能相距较远的类中心
def randCent(dataSet,k):
    m,n = dataSet.shape #m为行，n为列
    centroids = np.zeros((k,n))
    index = np.random.randint(0, m)
    centroids[0,]=dataSet[index,];

    # 2、初始化一个距离的序列
    d = [0.0 for _ in range(m)]
    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            # 3、对每一个样本找到最近的聚类中心点
            d[j] = nearest(dataSet[j, ], centroids[0:i, ])
            # 4、将所有的最短距离相加
            sum_all += d[j]
        # 5、取得sum_all之间的随机值
        sum_all *= random.rand()
        # 6、获得距离最远的样本点作为聚类中心点
        for j, di in enumerate(d):
            sum_all=sum_all - di
            if sum_all > 0:
                continue
            centroids[i,] = dataSet[j, ]
            break
    return centroids

def kmeans(dataSet,k):

    m=np.shape(dataSet)[0] #有多少行

    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.mat(np.zeros((m,2)))

    clusterChange = True

    # 第1步 初始化centroids 初始化k个随机的质心
    centroids = randCent(dataSet,k);
    while clusterChange:
        clusterChange=False

         # 遍历所有的样本（行数）
        for i in range(m):
            minDist=MAX;
            minIndex=MININDEX;

             # 遍历所有的质心
            #第2步 找出距离第i个样本最近的质心
            for j in range(k):
                # 计算该样本到质心的欧式距离
                distance=distEclud(centroids[j,:],dataSet[i,:])
                if distance<minDist:
                    minDist=distance
                    minIndex=j
            # 第 3 步：更新每一行样本所属的簇
            if clusterAssment[i,0] !=minIndex:
                clusterChange=True;#还有的更新图像还没有趋于稳定，继续循环
                clusterAssment[i,:] = minIndex,minDist**2 #更新
        #第 4 步：更新质心
        for j in range(k):
            # 获取簇类所有的点
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:,0].A == j)[0]] 
            # 对矩阵的行求均值 更新质心
            centroids[j,:] = np.mean(pointsInCluster,axis=0)   # 对矩阵的行求均值


    return clusterAssment.A[:,0], centroids

#生成k-means聚类测试用数据
def create_data_set(*cores):
    ds=list();
    for x0,y0,z0 in cores:
        x = np.random.normal(x0, 0.1+np.random.random()/3, z0)
        y = np.random.normal(y0, 0.1+np.random.random()/3, z0)
        ds.append(np.stack((x,y), axis=1))

    return np.vstack(ds)

if __name__ == "__main__":
    k=4
    ds = create_data_set((0,0,2500), (0,2,2500), (2,0,2500), (2,2,2500))

    #测试时间
    t0 = time.time()
    result, cores = kmeans(ds, k)
    t = time.time() - t0

    #画图像
    plt.scatter(ds[:,0], ds[:,1], s=1, c=result.astype(np.int))
    plt.scatter(cores[:,0], cores[:,1], marker='x', c=np.arange(k))
    plt.show()

    print("算法用时：",t);
