#Mean shift算法，又称均值漂移算法，这是一种基于核密度估计的爬山算法，
# 可用于聚类、图像分割、跟踪等。它的工作原理基于质心，
# 这意味着它的目标是定位每个簇/类的质心，即先算出当前点的偏移均值，
# 将该点移动到此偏移均值，然后以此为新的起始点，继续移动，
# 直到满足最终的条件（找出最密集的区域）。

#1.为了理解均值漂移，我们可以像上图一样想象二维空间中的一组数据点，然后先随机选择一个点C，以它为圆心画一个半径为r的圆开始移动。之前提到了，这是个爬山算法，它的核函数会随着迭代次数增加逐渐向高密度区域靠近。
#2.在每轮迭代中，算法会不断计算圆心到质心的偏移均值，然后整体向质心靠近。漂移圆圈内的密度与数据点数成正比。到达质心后，算法会更新质心位置，并继续让圆圈向更高密度的区域靠近。
#3.当圆圈到达目标质心后，它发现自己无论朝哪个方向漂移都找不到更多的数据点，这时我们就认为它已经处于最密集的区域。
#4.这时，算法满足了最终的条件，即退出。

# https://blog.csdn.net/gdkyxy2013/article/details/95733296

import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.lib.shape_base import dsplit

# 欧氏距离计算
def distEclud(pointA, pointB):
    '''
    计算欧氏距离
    :param pointA: A点坐标
    :param pointB: B点坐标
    :return: 返回得到的欧氏距离
    '''
    return math.sqrt((pointA - pointB) * (pointA - pointB).T)

def gs_kernel(dist,h):
    '''
    高斯核函数
    :param dist: 欧氏距离
    :param h: 带宽 
    h为带宽，当带宽一定时，样本点之间的距离越近，核函数的值越大；当样本点距离一定时，带宽越大，核函数的值越小。
    :return: 返回高斯核函数的值
    '''
    m=np.shape(dist)[0] #样本个数
    one = 1/(h*math.sqrt(2*math.pi))#画的圆    
    two = np.mat(np.zeros((m, 1)))
    for i in range(m):
        two[i, 0] = (-0.5 * dist[i] * dist[i].T) / (h * h)
        two[i, 0] = np.exp(two[i, 0])
    
    gs_val = one * two
    return gs_val

def shift_point(point, points, h):
    '''
    计算漂移向量
    :param point: 需要计算的点
    :param points: 所有的样本点
    :param h: 带宽
    :return: 返回漂移后的点
    '''
    points = np.mat(points)
    m = np.shape(points)[0]  # 样本的个数
    #计算距离
    point_dist = np.mat(np.zeros((m, 1)))
    for i in range(m):
        point_dist[i, 0] = distEclud(point, points[i])
    
    # 计算高斯核函数
    point_weights = gs_kernel(point_dist, h)

    # 计算分母
    all_sum = 0.0
    for i in range(m):
        all_sum += point_weights[i, 0]
        
    # 计算均值偏移
    point_shifted = point_weights.T * points / all_sum
    return point_shifted

def lb_points(mean_shift_points):
    '''
    计算所属类别
    :param mean_shift_points: 漂移向量
    :return: 返回所属的类别
    '''
    lb_list = []
    m, n = np.shape(mean_shift_points)
    index = 0
    index_dict = {}
    for i in range(m):
        item = []
        for j in range(n):
            item.append(str(("%5.2f" % mean_shift_points[i, j])))
        item_1 = "_".join(item)
        if item_1 not in index_dict:
            index_dict[item_1] = index
            index += 1

    for i in range(m):
        item = []
        for j in range(n):
            item.append(str(("%5.2f" % mean_shift_points[i, j])))
 
        item_1 = "_".join(item)
        lb_list.append(index_dict[item_1])
    return lb_list

def mean_shift(points, h=2, MIN_DISTANCE=0.000001):
    '''
    训练Mean Shift模型
    :param points: 特征点
    :param h: 带宽
    :param MIN_DISTANCE: 最小误差
    :return: 返回特征点、均值漂移点、类别
    '''
    mean_shift_points=np.mat(points)
    max_min_dist = 1;
    iteration=0; # 迭代的次数
    m = np.shape(mean_shift_points)[0]  # 样本的个数
    need_shift = [True] * m  # 标记是否需要漂移

    # 计算均值漂移向量
    while max_min_dist>MIN_DISTANCE:
        max_min_dist = 0;
        iteration+=1;
        print("iteration : " + str(iteration))
        for i in range(0,m):
            if not need_shift[i]:  # 判断每一个样本点是否需要计算偏移均值
                continue
            point_new = mean_shift_points[i]
            point_new_start = point_new
            point_new  = shift_point(point_new, points, h)  # 对样本点进行漂移计算
        
            dist = distEclud(point_new, point_new_start)  # 计算该点与漂移后的点之间的距离
            if dist>max_min_dist:
                max_min_dist=dist;
            if dist<MIN_DISTANCE:
                need_shift[i] = False;
            mean_shift_points[i] = point_new
    
    # 计算最终的类别
    lb = lb_points(mean_shift_points)  # 计算所属的类别
    return np.mat(points), mean_shift_points, lb

#生成数据
def create_data_set(*cores):
    ds=list();
    for x0,y0,z0 in cores:
        x = np.random.normal(x0, 0.1+np.random.random()/3, z0)
        y = np.random.normal(y0, 0.1+np.random.random()/3, z0)
        ds.append(np.stack((x,y), axis=1))

    return np.vstack(ds)

if __name__ == "__main__":
    ds = create_data_set((0,0,100), (0,2,100), (2,0,100), (2,2,100))
    points, shift_points, cluster = mean_shift(ds, 2)
    #print(points)
    #print(shift_points)
    print(cluster)
    #plt.scatter(ds[:,0], ds[:,1], s=1, c=cluster.astype(np.int))
    #plt.scatter(cores[:,0], cores[:,1], marker='x', c=np.arange(k))
    #plt.show()


