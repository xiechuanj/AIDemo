# -*- coding: utf-8 -*-

import copy
import numpy as np

class OutputNet:

    def __init__(self):
        self.weights       = []
        self.output_value  = 0.0  # 网路输出值，因为是做线性组合，故网路输出值即为外部 Hidden Layer Nets 的 sum(z(j) * wj)
        self.output_error  = 0.0  # 输出误差值（外部算完给进来）， Uses it to update weights, centers, sigmas, not cost_value.

    # random_count: 上一层要连进来的权重数量
    def randomize_weights(self, random_count=1, min=-0.25, max=0.25):
        self.clear_weights()
        for i in range(0, random_count):
            self.weights.append(np.random.uniform(min, max))

    def zero_weights(self, count=1):
        self.randomize_weights(count, 0.0, 0.0)

    def setup_weights(self, weights=[]):
        if not weights:
            return
        self.weights = copy.deepcopy(weights)

    def clear_weights(self):
        del self.weights[:]

    def refresh_weights(self, new_weights=[]):
        if not new_weights:
            return
        self.clear_weights()
        self.weights = copy.deepcopy(new_weights)

    def output(self, rbf_values=[]):
        if not rbf_values:
            return -1.0
        value = 0.0
        for index, net_weight in enumerate(self.weights):
            rbf_value = rbf_values[index]
            value     += (rbf_value * net_weight)
        self.output_value = value
        return  value
