# -*- coding: utf-8 -*-

import numpy as np
from distance import Method, Distance

class Group:

    def __init__(self, center=[], tag=""):
        self.samples            = []         # 本群分类进来的样本点
        self.center             = center     # 群心
        self.old_center         = []         # 旧的群心
        self.distance           = Distance()
        self.distance_method    = Method.Eculidean
        self.tag                = tag

    def add_sample(self, sample):
        if not sample:
            return
        self.samples.append(sample)

    # 更新群心
    def update_center(self):
        # e.g.
        # samples = [x1, x2, x3]
        # x1 = [2, 2, 3, 3]
        # x2 = [4, 5, 6, 7]
        # x3 = [7, 8, 9, 1]
        #
        # summed_vectors = [(2+4+7), (2+5+8), (3+6+9), (3+7+1)]
        #                = [13, 15, 18, 11]
        #
        # new_center     = [13, 15, 18, 11] / len(samples)
        #                = [13, 15, 18, 11] / 3
        #                = [4.33333333, 5.0, 6.0, 3.66666667]

        # 把所有所属的样本点做矩阵相加 （相同维度位置彼此相加), summed_vectors 会是 numpy.ndarray 型态
        summedd_vectors = np.sum(self.samples, axis=0)
        # 再把加总起来的矩阵值除以所有样本数，取得平均后的新群心的特征属性， new_center 也是 numpy.ndarray型态
        new_center = summedd_vectors / float(len(self.samples))
        # 更新新旧 Center 的记录 （要把 new_center 转回去 python 的List)
        self.old_center = self.center
        self.center     = new_center.tolist()

    # 清空记录的样本点
    def clear(self):
        # Romoving reference in this example is enough.
        del self.samples[:]

    # 计算本群里所有样本点对群心的距离总和，即为本群的 SSE 值
    def sse(self):
        if not self.samples:
            return 0.0
        sum = 0.0
        for point in self.samples:
            sum += self.distance.calculate(point, self.center, self.distance_method)
        return sum

    # 新旧群心的变化距离
    def diff_distance(self):
        if not self.old_center:
            return 0.0
        return self.distance.calculate(self.center, self.old_center, self.distance_method)
