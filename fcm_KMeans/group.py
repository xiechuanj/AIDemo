# -*- coding: utf-8 -*-

import numpy as np
from distance import Method, Distance

class Group:

    def __init__(self, center=[], tag=""):
        self.samples            = []         # 本群分类进来的样本点 (跟 all_memberships 没有索引位置上的对应关系)
        self.all_memberships    = []         # 所有样本点对本群心的归属度
                                             # FCM 是整体样本在做估计和运算（含更新群心、SSE)，不像 K-Means 是个体群在做运算和更新群心
        self.center             = center     # 群心
        self.old_center         = []         # 旧的群心
        self.distance           = Distance()
        self.distance_method    = Method.Eculidean
        self.tag                = tag
        self.sse                = 0.0        # 本群 SSE （由外部提供， 這里做記錄）

    def add_sample(self, sample):
        if not sample:
            return
        self.samples.append(sample)

    def add_membership(self, membership=1.0):
        self.all_memberships.append(membership)

    # 要用全部的样本点（all_samples)来更新群心
    # all_samples 会对应 self.all_memberships的索引位置
    def update_center(self, all_samples=[], m=2):
        # 把所有的样本点先各自乘上自己对本群心的归属度 m 次方后，再做矩阵相加（相同维度位置彼此相加）。
        # membership_samples 和 summed_vectors 都会是 numpy.ndarray 型态
        length          = len(all_samples[0])
        summed_vectors = np.zeros(length) # 先建一个跟样本点特征长度等长的零值阵列
        for index, point in enumerate(all_samples):
            # 取出该样本点对本群的归属度
            membership          = self.all_memberships[index]
            # 归属度 m 次方
            m_membership        = membership ** m
            membership_point    = np.multiply(point, m_membership)
            # 加总
            summed_vectors      = np.add(membership_point, summed_vectors)

        # 再把加总起来的 summed_vectors 矩阵值除以 "所有样本点对本群心的归属度 m 次方总和"，
        # new_center 也是 numpy.ndarray 形态
        summed_membership  = 0.0
        for membership in self.all_memberships:
            summed_membership += (membership ** m)
        new_center = summed_vectors / summed_membership

        # 更新新旧 Center 的记录（要把 new_center 从 numpy 物件转回去 python 的 List)
        self.old_center   = self.center
        self.center       = new_center.tolist()

    # 清空记录的样本点
    def clear(self):
        del self.samples[:]
        del self.all_memberships[:]
        self.sse = 0.0

    # 新旧群心的变化距离
    def diff_distance(self):
        if not self.old_center:
            return 0.0
        return self.distance.calculate(self.center, self.old_center, self.distance_method)

