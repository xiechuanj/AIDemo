# -*- coding: utf-8 -*-

import copy

class CenterNet:

    def __init__(self, features=[]):
        self.features  = copy.deepcopy(features)  # 中心点自己的特征向量
        self.index     = 0                        # 第几笔资料或第几颗神经元
        self.sigma     = 1.0
        self.sum_singal = 0.0                     # ||X(p) - Cj(P)||, the center net sums features of pattern to center the distance.
        self.rbf_value  = 0.0                     # The RBF value of this center of current pattern.

    def refresh_features(self, new_features=[]):
        if not new_features:
            return
        self.features = copy.deepcopy(new_features)
