# -*- coding: utf-8 -*-

import numpy as np

class Activation:

    # 2-norm without sqrt, ||x1 - x2||^2
    def eculidean_without_sqrt(self, x1=[], x2=[]):
        sum = 0.0
        for i, a in enumerate(x1):
            b = x2[i]
            sum += (a - b) ** 2
        return sum

    # 2-norm
    def eculidean(self, x1=[], x2=[]):
        sum = self.eculidean_without_sqrt(x1, x2)
        return np.sqrt(sum)

    def activate(self, input_features=[], center_features=[], center_sigma=1.0):
        # RBF: exp(||x - cj||^2 / (2 * sigma^2))
        net_input = self.eculidean_without_sqrt(input_features, center_features)
        return np.exp(-net_input / (2.0 * (center_sigma ** 2)))

    def partial(self, net_output=0.0, center_sigma=1.0):
        return -((2.0 * net_output) / (2.0 * (center_sigma ** 2))) * np.exp(-net_output / (2.0 * (center_sigma ** 2)))