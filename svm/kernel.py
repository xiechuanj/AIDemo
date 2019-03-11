# -*- coding: utf-8 -*-
from enum import Enum
import numpy as np


class Method(Enum):
    Linear  = 0
    RBF     = 1  # 径向基（高斯）
    Sigmoid = 2  # 余弦相似
    Tanh    = 3

class Kernel:

    def __init__(self, method=Method.Linear):
        self.method = method
        self.sigma  = 2.0      # RBF 用的标准差
        self.slope  = 1.0      # Sigmoid, Tanh 用的坡度值

    def calculate(self, x1=[], x2=[]):
        if not x1 or not x2:
            return
        return {
            Method.Linear: self.linear,
            Method.RBF: self.rbf,
            Method.Sigmoid: self.sigmoid,
            Method.Tanh: self.tanh
        }.get(self.method)(x1, x2)

    def linear(self, x1, x2):
        return np.dot(x1, x2)

    def rbf(self, x1, x2):
        s = 0.0
        for index, x1_feature in enumerate(x1):
            # Formula: s += (x1[i] - x2[i])^2
            x2_feature = x2[index]
            s += (x1_feature - x2_feature) ** 2
        # RBF Formula: exp^(-s / (2.0 * sigma * sigma))
        return np.exp(-s / (2.0 * (self.sigma ** 2)))

    def sigmoid(self, x1, x2):
        x = self.linear(x1, x2)
        return 1.0 / (1.0 + np.exp(-self.slope * x))

    def tanh(self, x1, x2):
        x = self.linear(x1, x2)
        return 2.0 / (1.0 + np.exp(-self.slope * x)) - 1.0

