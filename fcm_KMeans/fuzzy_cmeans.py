# -*- coding: utf-8 -*-

import numpy as np
import copy
from distance import Method, Distance
from group import Group

class FuzzyCMeans:

    __iteration_times = 0 # 迭代数 （已训练次数)

    def __init__(self):
        self.samples             = []                  # 训练样本集： [features array of each sample]
        self.distance_method     = Method.Eculidean    # 要用什么距离算法
        self.distance            = Distance()          # 距离算法
        self.groups              = []                  # 每一个分类好的群聚 [Group]
        self.convergence         = 0.001               # 收敛误差
        self.max_iteration       = 100                 # 最大迭代次数
        self.m                   = 2                   # 计算归属度的参数 （m 是模糊器， 用来决定每一个群聚的模糊程度，m 太小则算法效果会接近硬聚类，太大则效果差）
        self.iteration_callback  = None
        self.completion_callback = None

    def add_sample(self, features=[]):
        if not features:
            return
        self.samples.append(copy.deepcopy(features))    # Deep copy features (point).

    # 外部直接呼叫给中心点的话，就是专家自订中心点
    def add_center(self, center=[], tag=""):
        if not center:
            return
        # 建立要分类的群
        self.groups.append(Group(center, tag))

    # 这里演示的中心点选取方法是随机挑
    def shuffing_pick(self, k=1):
        if not self.samples:
            return
        # 选到中心点时，会同时做成 Group
        # 先做一个同样本数据的 0 ~ n连续索引表，再打乱这索引表来做中心点的选择
        random_indexes = np.arange(len(self.samples))
        np.random.shuffle(random_indexes)   # 打乱
        for i in range(k):
            random_index = random_indexes[i]
            centroid     = self.samples[random_index]
            tag = "group_" + str(len(self.groups)+1)
            self.add_center(center, tag)

    # 清空每一群里已记录的样本点
    def clear_groups(self):
        for group in self.groups:
            group.clear()

    # 更新群心
    def update_groups(self):
        for group in self.groups:
            group.update_center(self.samples, self.m)

    # 在执行 run() 之前先跑一次 setup() 把其它需要初始的参数做一个设定
    def setup(self):
        for group in self.groups:
            group.distance_method = self.distance_method

    def run(self, iteration_callback, completion_callback):
        self.__iteration_times   = 0
        self.iteration_callback  = iteration_callback
        self.completion_callback = completion_callback
        self._training()

    # 分类（预测）新的数据点
    def classify(self, points, completion_callback):
        if not points:
            return None
        for point in points:
            to_group = self._nearest_group(point)
            if completion_callback:
                completion_callback(point, to_group)

    # SSE, 计算所有样本点对所有群的（该点对群的归属度 * 该点对该群心的距离）总和，即为本次聚类的 SSE 值
    # "SSE 也是所有样本在运算，不是个别拥有的样本点在运算 （跟 K-Means 的 SSE 模式不同）"
    def sse(self):
        if not self.samples:
            return  0.0

        sum_sse = 0.0
        # 先算所有样本点对所有群的距离
        for point in self.samples:
            all_distances = self._point_all_distances(point)
            # 取出自己对第 k 群的距离，
            for k, self_distance in enumerate(all_distances):
                # 再累加自己的对第 k 群的距离 除以 对其它群的距离比例
                sum_ratio = 0.0
                for other_distance in all_distances:
                    if other_distance > 0.0:
                        sum_ratio += (self_distance / other_distance)
                membership_degree = 1 / (sum_ratio ** (2 / (self.m - 1)))

                # FCM 的 SSE 公式是：SUM（归属度 * 群心距离）:
                # a. 计算个群的 SSE
                group          = self.groups[k]
                group_sse      = membership_degree * self_distance
                group.sse      += group_sse

                # b.计算总合的 SSE
                sum_sse += group_sse

        return sum_sse


    '''
    @ Private
    '''
    # 计算指定样本点对所有群的距离
    def _point_all_distances(self, point=[]):
        all_distances = []
        for group in self.groups:
            centroid  = group.center
            distance  = self.distance.calculate(point, centroid, self.distance_method)
            # 原公式会对距离再做一次平方
            # 再提醒： 演算法是很活的，最重要的是它的结构与运作概念，这里即使直接使用欧式或者其它距离公式都行
            # all_distances.append(distance ** 2)
            all_distances.append(distance)
        return all_distances

    # 被归类到哪群（最近的群， 找最大的归属度）
    def _nearest_group(self, point):
        # 计算归属度 （Membership):
        # 1. 先把当前的 point 点对每一个群心做距离运算
        # 2. 把距离都先放入 array 里
        all_distances = self._point_all_distances(point)

        # 3. 把距离都取出来照顺序运算归属度
        to_index        = -1 #被分到哪一群
        max_membership  = -1
        # 先取出自己对第 k 群里的距离
        for k, self_distance in enumerate(all_distances):
            # 再累加自己的对第 k 群的距离 除以 对其它群的距离比例
            sum_ratio = 0.0
            for other_distance in all_distances:
                if other_distance > 0.0:
                    sum_ratio += (self_distance / other_distance)
            # 算法视频讲 1/(m-1), 而有的文献则是惯用 2/(m-1), 演算方法是很活的， 没有一定要怎么做的限制。
            # 而这里 Coding 的部分，考量到也许大多数使用到 Fuzzy C-Means 的 Library 会偏向采用 2/(m-1) 的关系
            # 我们这里也采用该惯用作法。
            membership_degree = 1 / (sum_ratio ** (2 /(self.m - 1)))
            # 把样本对该 k 群的归属度记起来（更新群心用）
            group = self.groups[k]
            group.add_membership(membership_degree)

            # 4. 取最大归属度并分到该群
            if membership_degree > max_membership:
                to_index    = k
                max_membership = membership_degree
        # 5. 回传该 point 点被分进去的群
        return self.groups[to_index]

    # 训练
    def _training(self):
        self.__iteration_times += 1
        # 清空每群里的样本
        self.clear_groups()
        for point in self.samples:
            # 分进去的群
            to_group = self._nearest_group(point)
            # 分进去所属的群
            to_group.add_sample(point)

        # 更新群心
        self.update_groups()

        # 计算是否已达到收敛条件
        changed_distance = -1.0
        for group in self.groups:
            diff_distance = group.diff_distance()
            if diff_distance > changed_distance:
                changed_distance = diff_distance

        # 新旧群心的变化量 <= 收敛误差 or 迭代数达到最大迭代限制
        if (changed_distance <= self.convergence) or (self.__iteration_times >= self.max_iteration):
            # 停止运算
            if self.completion_callback:
                self.completion_callback(self.__iteration_times, self.groups, self.sse())
        else:
            # 继续运算
            if self.iteration_callback:
                self.iteration_callback(self.__iteration_times, self.groups)
            self._training()


