# -*- coding: utf-8 -*-

import numpy as np
import copy
from kernel import Method, Kernel

# The object of training sample
class Sample:

    def __init__(self, features=[], target_value=0.0):
        self.features         = copy.deepcopy(features)     # 样本的特征向量
        self.target_value     = target_value                # e.g. +1, -1
        self.label            = target_value                # e.g. +1, -1， 0， 2， 3， (使用多分类）
        self.alpha_value      = 0.0
        self.error_value      = 0.0
        self.tolerance_error  = 0.001
        self.index            = 0                           # Outside index number within samples of parent-class
        self.kernel           = Kernel(Method.Linear)       # 预设使用线性分割（Linear)

    # return BOOL， 检查目前的样本点是否符合 KKT条件
    def is_confom_kkt(self, samples=[], bias=0.0, const_value=0.0):
        sum_x    = - bias
        for sample_x in samples:
            if sample_x.alpha_value != 0:
                sum_x += sample_x.alpha_value * sample_x.target_value * self.kernel.calculate(sample_x.features, self.features)

        kkt_value = self.target_value * (sum_x)
        # 进行KKT 条件判断（有3个KKT满足条件）
        alpha_value      = self.alpha_value
        tolerance_error  = self.tolerance_error
        is_conformed     = True
        if alpha_value == 0.0 and (kkt_value + tolerance_error) >= 1.0:
            pass
        elif alpha_value == const_value and (kkt_value - tolerance_error) <= 1.0:
            pass
        elif alpha_value > 0.0 and alpha_value < const_value and abs(kkt_value - 1.0) <= tolerance_error:
            pass
        else:
            is_conformed = False
        return is_conformed