# -*- coding: utf-8 -*-

import numpy as np
from enum import Enum
from kmeans.distance import Method, Distance

'''
中心点选取方法
'''
class Choice(Enum):
    Shuffing = 0 # 随机挑
    Make     = 1 # 随机给中心点
    Plus     = 2 # K-Means++

class Maker:

    def __init__(self):
        self.samples         = []
        self.distance_method = Method.Eculidean
        self.distance        = Distance()

    # 选中心点的方法
    # 1. 随机挑（从现有的数据里随意挑 K 个中心点
    # 2. 随机给（每个中心点的特征值都 是随机生成的，不从现有数据里挑选）
    # 3. 优化挑（K-Means++)
    # 4. 专家给（自订，外部直接给)

    # 随机挑（乱数）
    def shuffing(self, k):
        centers     = []
        random_list = np.arange(len(self.samples))  #先建一个同 samples array 长度的乱序索引阵列
        np.random.shuffle(random_list)  # 打乱
        for i in range(k):
            random_index = random_list[i]
            centers.append(self.samples[random_index])
        return centers

    # 随机给中心点
    def make(self):
        centroids = []
        dimension = len(self.samples)  # 有几维
        points = np.array(self.samples) # 转成numpy的array在这里比较好操作
        # 这里不把 k 的圈里去让center 存 random_feature 的值
        # 是国为这样高速产生出来的随机值才不会太相近（比较离散），虽然这么做会多耗去一点在重复取 min, max 的效能。
        for i in xrange(k):
            center = []
            for n in xrange(dimension):
                # 1. 取出 samples 训练样本集里，同 n 维度的所有样本的维度集合
                # 2. 即时运算该 n 维度集合里的 min, max 特征值
                # 3. 制作中心点
                # e.g.
                #   samples = [[1, 2, 3, 3], [4, 5, 6, 7], [7, 8, 9, 1]]
                #   array([1, 4, 7]), min 1 max 7
                #   array([2, 5, 8]), min 2 max 8
                #   array([3, 6, 9]), min 3 max 9
                #   array([3, 7, 1]), min 1 max 7
                dimension_samples = points[:n]
                min = np.min(dimension_samples)
                max = np.max(dimension_samples)
                # Random formuls: (b -a) * random_sample() + a
                random_feature = (max - min) * np.random.random_sample() + min
                center.append(random_feature)
            centroids.append(center)
        return centroids

    # 优化挑 （K-Means++)
    def plus(self, k):
        # 1. 随机挑 1 个样本点当初始群心 （k=1), 挑好后把 k 减 1
        centers = self.shuffing(1)
        k   -= 1
        # 2. 找每个样本最近的群心距离累加起来， 得到 S (这里是： sum_shortest)
        max_float = float("inf")
        for c in range(k):
            sum_shortsest      = 0.0
            shortest_distances = []  # 把最短距离放入阵列里记起来，阵列的索引位置等同于 samples 的索引位置
            for point in self.samples:
                min_distance = max_float  # 找最小的距离
                # 找最小距离的时候，如果样本所对上的是被挑到当群心的自己，则距离会为 0， 而在累加和连减时并不会影响整个运算结果
                for centroid in centers:
                    distance = self.distance.calculate(point,centroid, self.distance_method)
                    if distance < min_distance:
                        min_distance = distance
                sum_shortsest += min_distance
                shortest_distances.append(min_distance)
            # 3. 把累加距离乘上 （0.0， 1.0] 之间的随机值，得到 Srandom (这里是： random_sum)
            # 附注： numpy 的 random 是 [0.0， 1.0），而我们想要的随机值是（0.0，1.0] 要能 > 0.0 并包含 1.0
            random_value = np.random.random_sample()
            # 判断随机值如果为 0， 就直接返回 1
            if random_value == 0.0:
                random_value = 1.0
            random_sum = sum_shortsest * random_value
            # 4. 用Srandom (random_sum) 依序连减这些最短距离值， 直到 <= 0.0
            for index, distance in enumerate(shortest_distances):
                random_sum -= distance
                # 5. 取出减到为 0 时的样本点， 即为下一个群心
                if random_sum <= 0.0:
                    next_centroid = self.samples[index]
                    centers.append(next_centroid)
                    break
            # 6. 如未挑满 k 个群心，即重复步骤 2 ~ 4
        return centers

    '''
    samples: 训练样本集
    method: 挑选中心点的方法
    k: 要分几群    
    '''
    def centers(self, samples=[], method=0, k=1):
        self.samples =samples
        return {
            Choice.Shuffing: self.shuffing,
            Choice.Make: self.make,
            Choice.Plus: self.plus
        }.get(method)(k)



