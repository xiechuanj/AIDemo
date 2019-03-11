# -*- coding: utf-8 -*-

import numpy as np
from distance import Method, Distance
from center import Choice, Maker
from group import Group

class KMeans:

    __iteration_times = 0 # 迭代数 （已训练次数)

    def __init__(self):
        self.samples             = []
        self.distance_method     = Method.Eculidean
        self.distance            = Distance()
        self.center_choice       = Choice.Shuffing
        self.center_maker        = Maker()
        self.groups              = []
        self.convergence         = 0.001
        self.max_iteration       = 100
        self.iteration_callback  = None
        self.completion_callback = None

    def add_sample(self, features=[]):
        if not features:
            return
        self.samples.append(features)

    # 外部直接呼叫给中心点的话，就是专家自订中心点
    def add_center(self, center=[], tag=""):
        if not center:
            return
        # 建立要分类的群
        self.groups.append(Group(center, tag))

    def make_centers(self, k=1):
        if not self.samples:
            return
        # 选完的中心点会再做成一个一个的 Group
        created_centers = self.center_maker.centers(self.samples, self.center_choice, k)
        for center in created_centers:
            # 给每个群预设权重
            tag = "group_" + str(len(self.groups)+1)
            self.add_center(center, tag)

    # 清空每一群里已记录的样本点
    def clear_groups(self):
        for group in self.groups:
            group.clear()

    # 更新群心
    def update_groups(self):
        for group in self.groups:
            group.update_center()

    # 在执行 run() 之前先跑一次 setup() 把其它需要初始的参数做一个设定
    def setup(self):
        for group in self.groups:
            group.distance_method = self.distance_method

    # 开始训练
    def run(self, iteration_callback, completion_callback):
        self.__iteration_times   = 0
        self.iteration_callback  = iteration_callback
        self.completion_callback = completion_callback
        self._training()

    # 分类（预测）新的数据点
    def classify(self, points, completion_callback):
        if not points:
            return None
        max_float = float("inf")
        for point in points:
            to_group = self._nearest_group(point, max_float)
            if completion_callback:
                completion_callback(point, to_group)

    # SSE, 加总所有的分群里头每个资料点与中心点的距离
    # 用于对每次 K-Means 的聚类结果做评量，以找出具有最小 SSE 的那组聚类结果作为解答。
    def sse(self):
        result_sse = 0.0
        for group in self.groups:
            result_sse += group.sse()

        return result_sse

    '''
    @ Private
    '''
    # 被归类哪群（最近的群）
    def _nearest_group(self, point, max_float):
        to_index    = -1   # 被分到哪一群
        min_distance = max_float # 找最小的距离
        for index, group in enumerate(self.groups):
            centroid = group.center
            distance = self.distance.calculate(point, centroid, self.distance_method)
            if distance < min_distance:
                to_index = index
                min_distance = distance
            else:
                continue
        return self.groups[to_index]

    # 训练
    def _training(self):
        self.__iteration_times += 1
        # 清空每群里的样本
        self.clear_groups()
        max_float = float("inf")
        for point in self.samples:
            to_group = self._nearest_group(point, max_float)
            # 分进去所属的群
            to_group.add_sample(point) # Uses reference not deeply copying the point object.

        # 更新群心
        self.update_groups()

        # 计算是否已达到收敛条件
        #    先算所有群心的新旧距离变化是否已达到收敛标准 （接近一再变化），有 3 种对变化量对收敛误差的判断
        #         1. 取最大变化值
        #         2. 取最小变化值
        #         3. 取平均变化值
        #    这里的范例将使用 3。 取平均变化量
        sum_distance = 0.0
        for group in self.groups:
            sum_distance += group.diff_distance()
        changed_distance = sum_distance / len(self.groups)
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


