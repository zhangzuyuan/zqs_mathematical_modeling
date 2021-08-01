'''
MDS是使得降维后的两点间的实际范数与理想距离最近（图论中是两点间最短路是理想距离）
本来表示这一组数据需要每两个点之间互相对应
而现在只需要存n维数据每两个点之间距离可以用这两个点之间n维范数来表示
所以这个降维方法可以用于图布局，降成二维，两个维度即为x与y坐标
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
MAX=float('inf')
#读取数据，数据为图论数据
def readGraph(path):
    data=np.loadtxt(path,dtype='int', delimiter=' ');
    return data

#转换成矩阵
def maxtrixGraph(data):
    maxPoint=0;
    for item in data:
        maxPoint=max(maxPoint,max(item[0],item[1]));
    print(maxPoint);
    matrix=np.zeros([maxPoint+1,maxPoint+1]); 
    for i in range(maxPoint+1):
        for j in range(maxPoint+1):
            if i==j:
                matrix[i][j]=0;
            else:
                matrix[i][j]=MAX;
    for item in data:
        matrix[item[0]][item[1]]=1;
        matrix[item[1]][item[0]]=1;
    return matrix

#求两点间最短路，目标是使得降维得到的两点间数据的范数与理想距离（两点间最短路）差别最小
def dijkstra(matrix, start_node):
    
    #矩阵一维数组的长度，即节点的个数
    matrix_length = len(matrix)

    #访问过的节点数组
    used_node = [False] * matrix_length

    #最短路径距离数组
    distance = [MAX] * matrix_length

    #初始化，将起始节点的最短路径修改成0
    distance[start_node] = 0
    
    #将访问节点中未访问的个数作为循环值，其实也可以用个点长度代替。
    num=0;
    while used_node.count(False):
        num=num+1;
        if num>matrix_length:
            break;
        min_value = float('inf')
        min_value_index = 9999
        
        #在最短路径节点中找到最小值，已经访问过的不在参与循环。
        #得到最小值下标，每循环一次肯定有一个最小值
        for index in range(matrix_length):
            if not used_node[index] and distance[index] < min_value:
                min_value = distance[index]
                min_value_index = index
        
        #将访问节点数组对应的值修改成True，标志其已经访问过了
        if min_value_index!=9999:
            used_node[min_value_index] = True

        #更新distance数组。
        #以B点为例：distance[x] 起始点达到B点的距离，
        #distance[min_value_index] + matrix[min_value_index][index] 是起始点经过某点达到B点的距离，比较两个值，取较小的那个。
        for index in range(matrix_length):
            if min_value_index!=9999:
                distance[index] = min(distance[index], distance[min_value_index] + matrix[min_value_index][index])

    return distance

#幂迭代法 求最大和第二大特征值
def power_iteration(A,num:int):
    b_k = np.random.rand(A.shape[1]);
    for i in range(num):
        b_k1=np.dot(A,b_k);
        b_k1_norm=np.linalg.norm(b_k1);
        b_k = b_k1/b_k1_norm;
    return b_k;
    
#降维布局
def pivotMDS(matrix,k):
    n=len(matrix);
    dis=[];
    for i in range(len(matrix)):
        tmpd=dijkstra(matrix,i);
        dis.append(tmpd);
    for i in range(len(dis)):
        for j in range(len(dis)):
            if dis[i][j]==float('inf'):
                dis[i][j]=0;
    d=np.zeros([n,k]);
    d=np.asarray(dis);
    d=d[0:n,0:k];
    d2=d**2;
    deltaCol =d2.sum(axis=0)/n;
    deltaRow =d2.sum(axis=1)/k;
    sumALL = d2.sum()/(n*k);
    C = np.zeros([n,k]);
    for i in range(n):
        for j in range(k):
            d[i][j]
            C[i][j];
            deltaCol[j]
            deltaRow[i]
            C[i][j]=-(1.0/2)*(d[i][j]**2-deltaCol[j]-deltaRow[i]+sumALL);
    B=np.dot(C.T,C);
    V_1=power_iteration(B,100).reshape(1,-1);
    lbd=np.dot(V_1,np.dot(B,V_1.T));
    B_2=B-lbd/np.linalg.norm(V_1)**2*np.dot(V_1.T,V_1);
    V_2=power_iteration(B_2,100);
    ans=np.zeros([n,2]);
    ans[:,0] = np.dot(C,V_1.reshape(-1,1)).reshape(-1)
    ans[:,1] = np.dot(C,V_2.reshape(-1,1)).reshape(-1)
    return ans;

if __name__ == "__main__":
    file_name = 'text.txt' 
    data = readGraph(file_name);
    matrix=maxtrixGraph(data);
    ans = pivotMDS(matrix,50);
    print(ans);

    plt.figure(figsize=(8,8))
    plt.axis('equal')
    ax = plt.axes()
    ax.set_xlim(min(ans[:,0])-10, max(ans[:,0])+10)
    ax.set_ylim(min(ans[:,1])-10, max(ans[:,1])+10)

    lines = []
    for item in data:
        lines.append([ans[item[0]],ans[item[1]]]);
    lc = mc.LineCollection(lines, linewidths=.3, colors='#0000007f')
    ax.add_collection(lc)
    plt.show()
