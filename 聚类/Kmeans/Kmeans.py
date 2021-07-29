import numpy as np
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

# 为给定数据集构建一个包含K个随机质心的集合
#从数据的那几个点中随便挑选几个作为质心
def randCent(dataSet,k):
    m,n = dataSet.shape #m为行，n为列
    centroids = np.zeros((k,n))
    for i in range(k):
        index = int(np.random.uniform(0,m))
        centroids[i,:] = dataSet[index,:]
    return centroids

#k均值聚类
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
