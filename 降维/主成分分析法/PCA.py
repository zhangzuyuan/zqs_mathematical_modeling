#https://blog.csdn.net/u012162613/article/details/42177327

import numpy as np

#均值零化：将每个特征的均值都变成零
def zeroMean(dataMat):
    meanVal=np.mean(dataMat,axis=0);#按列求均值，即求每个特征的均值
    newData=dataMat-meanVal;
    return newData,meanVal;

def pca_n(dataMat,n):
    #求协方差矩阵
    newData,meanVal=zeroMean(dataMat);
    covMat=np.cov(newData,rowvar=0);

    #求特征值，特征矩阵
    eigVals,eigVects=np.linalg.eig(np.mat(covMat));

    #保留主要成分即包辣比较大的前几个特征值
    eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标
    n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量
    lowDDataMat=newData*n_eigVect               #低维特征空间的数据
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal  #重构数据
    return lowDDataMat,reconMat;

#有时候不知道n是多少，所以用降维前后方差百分比来代替确定降低到最合适的维度
def percentage2n(eigVals,percentage):
    sortArray=np.sort(eigVals)   #升序
    sortArray=sortArray[-1::-1]  #逆转，即降序
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum>=arraySum*percentage:
            return num

def pca(dataMat,percentage=0.99):
    newData,meanVal=zeroMean(dataMat)
    covMat=np.cov(newData,rowvar=0)    #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    n=percentage2n(eigVals,percentage)                 #要达到percent的方差百分比，需要前n个特征向量
    eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标
    n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量
    lowDDataMat=newData*n_eigVect               #低维特征空间的数据
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal  #重构数据
    return lowDDataMat,reconMat

if __name__ == "__main__":
    #newdata=pca_n(data,2);