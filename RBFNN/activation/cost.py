# -*- coding: utf-8 -*-

import numpy as np
import copy

class Cost(object):

    outputs = [] # <List <Double>>, 装每一个训练样本的输出值阵列
    targets = [] # <List <Double>>, 装每一个训练样本的目标值阵列

    def add(self, outputs=[], targets=[]):
        if not outputs or not targets:
            return
        self.outputs.append(copy.deepcopy(outputs))
        self.targets.append(copy.deepcopy(targets))

    def clear(self):
        del self.outputs[:]
        del self.targets[:]

    '''
    @ Private
    '''
    # 判断是否能计算MSE, RMSE
    def _can_calculate(self):
        return self.samples_count > 0 and self.outputs_count > 0

    '''
    @ Getters that all Readonly
    '''
    @property
    #代价植
    def cost_value(self):
        cost_value = 0.0
        # 计算每一个东西的 Cost Value
        for index, sample_outputs in enumerate(self.outputs):
            sample_targets = self.targets[index]
            # 取出 Output Layer 里每一颗的神经元输出 （Net Output)
            for net_index, net_output in enumerate(sample_outputs):
                net_target  = sample_targets[net_index]
                output_error = net_target - net_output
                cost_value += (output_error * output_error)

        return cost_value

    @property
    # 有几笔训练样本
    def samples_count(self):
        return len(self.outputs)

    @property
    # 有几个输出
    def outputs_count(self):
        return len(self.outputs[0])

    @property
    # 均方误差
    def mse(self):
        return (self.cost_value / (self.samples_count * self.outputs_count) * 0.5) if self._can_calculate() else float("inf")

    @property
    # 均方根误差
    def rmse(self):
        return np.sqrt(self.cost_value / (self.samples_count * self.outputs_count)) if self._can_calculate() else float("inf")

    @property
    # 交叉
    def cross_entropy(self):
        iteration_entropy = 0.0
        outputs_count   = self.outputs_count
        for index, sample_outputs in enumerate(self.outputs):
            sample_targets = self.targets[index]
            sample_entropy = 0.0
            for net_index, net_output in enumerate(sample_outputs):
                target_value  = sample_targets[net_index]
                entropy      = (target_value * np.log(net_output)) + ((1.0 - target_value) * np.log(1.0 - net_output))
                sample_entropy += entropy
            iteration_entropy += -(sample_entropy / outputs_count)
        return (iteration_entropy / self.samples_count)