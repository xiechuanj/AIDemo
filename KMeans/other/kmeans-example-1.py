# -*- coding: utf-8 -*-
# sklearn.cluster.KMeans(n_clusters=8,
#                        init='k-means++',
#                        n_init=10,
#                        max_iter=300,
#                        tol=0.0001,
#                        precompute_distances='auto',
#                        verbose=0,
#                        random_state=None,
#                        copy_x=True,
#                        n_jobs=1,
#                        algorithm='auto'
#                        )
#
# 参数的意义：
#
# n_clusters:簇的个数，即你想聚成几类
# init: 初始簇中心的获取方法
# n_init: 获取初始簇中心的更迭次数，为了弥补初始质心的影响，算法默认会初始10次质心，实现算法，然后返回最好的结果。
# max_iter: 最大迭代次数（因为kmeans算法的实现需要迭代）
# tol: 容忍度，即kmeans运行准则收敛的条件
# precompute_distances：是否需要提前计算距离，这个参数会在空间和时间之间做权衡，如果是True 会把整个距离矩阵都放到内存中，auto 会默认在数据样本大于featurs*samples 的数量大于12e6 的时候False,False 时核心实现的方法是利用Cpython 来实现的
# verbose: 冗长模式（不太懂是啥意思，反正一般不去改默认值）
# random_state: 随机生成簇中心的状态条件。
# copy_x: 对是否修改数据的一个标记，如果True，即复制了就不会修改数据。bool 在scikit-learn 很多接口中都会有这个参数的，就是是否对输入数据继续copy 操作，以便不修改用户的输入数据。这个要理解Python 的内存机制才会比较清楚。
# n_jobs: 并行设置
# algorithm: kmeans的实现算法，有：‘auto’, ‘full’, ‘elkan’, 其中 'full’表示用EM方式实现


import numpy as np
from sklearn.cluster import KMeans

# 生成一个随机数据，样本大小为100， 特征为3
data = np.random.rand(100, 3)

# 分群
estimator = KMeans(n_clusters=3)
# 聚类
estimator.fit(data)
# 获取聚类标签
label_pred = estimator.labels_
# 获取群心
centroids = estimator.cluster_centers_
# 获取聚类准则的总和
inertia = estimator.inertia_


# estimator初始化Kmeans聚类；estimator.fit聚类内容拟合；
# estimator.label_聚类标签，这是一种方式，还有一种是predict；estimator.cluster_centers_聚类中心均值向量矩阵
# estimator.inertia_代表聚类中心均值向量的总和
