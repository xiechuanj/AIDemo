# -*- coding: utf-8 -*-

from numpy import *

# 加载数据
def loadDataSet(fileName):  # 解析文件，按tab分割字段，得到一个浮点数字类型的矩阵
    dataMat = []      # 文件的最后一个字段是类别标签
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        # fltLine = map(float, curLine)  # 将每个元素转成float类型
        # print(fltLine)
        fltLine = list(map(float, curLine))  # 将每个元素转成float类型
        dataMat.append(fltLine)
    return dataMat

# 计算欧几里得距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA-vecB, 2)))  # 求两个向量之间的距离

# 构建聚簇中心，取k个（此例中为4）随机质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))   # 每个质心有n个坐标值，总共要k个质心
    for j in range(n):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
    return  centroids

# k-means聚类算法
def kMeans(dataSet, k, distMeans = distEclud, createCent = randCent):
    m = shape(dataSet)[0]
    # 用于存放样本属于哪类及质心距离
    clusterAssment = mat(zeros((m, 2)))
    # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
    centroids = createCent(dataSet, k)
    # 用于判断聚类是否已经收敛
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False;
        # 把每一个数据点划分到离它最近的中心点
        for i in range(m):
            minDist = inf; minIndex = -1;
            for j in range(k):
                distJI = distMeans(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    # 如果第i个数据点到第j个中心点，则将i归属为j
                    minDist = distJI;minIndex =j
            # 如果分配发生变化，则需要继续迭代
            if clusterAssment[i,0] != minIndex: clusterChanged = True;
            # 并将第i个数据点的分配情况存入字典
            clusterAssment[i,:] = minIndex, minDist**2

        print(centroids)
        # 重新计算中心点
        for cent in range(k):
            # 去第一列等于cent的所有列
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            # 算出这些数据的中心点
            centroids[cent,:] = mean(ptsInClust, axis = 0)
    return centroids, clusterAssment

# -----------------测试--------------------
# 用测试数据及测试kmeans算法
dataMat = mat(loadDataSet('testSet.txt'))
myCentroids, clustAssing = kMeans(dataMat, 4)
print(myCentroids)
print(clustAssing)





