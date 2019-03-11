# -*- coding: utf-8 -*-


import numpy as np


from activation import Activation

'''
单输出的Neuron
'''
class Neuron:
    # Private usage.
    __iteration_times = 0 # 迭代次数
    __iteration_error = 0.0 #迭代误差总和

    def __init__(self):
        self.tag              = self.__class__.__name__
        self.samples          = []   # 所有的训练样本（特征值)
        self.targets          = []   # 范本的目标输出
        self.weights          = []   # 权重
        self.bias             = 0.0  # 偏权值
        self.learning_rate    = 1.0  # 学习速率
        self.max_iteration    = 1    # 最大迭代数
        self.convergence      = 0.001# 收敛误差
        self.activation       = Activation()

    # Iteration Cost Function: 每个完整迭代运算后，把每一个训练样本的cost function取平均（用于判断是否收敛）
    def _iteration_cost_function(self):
        # 1/2 * (所有训练样本的cost function总和 / （训练样本数量 * 每笔训练样本的目标输出数量))
        return 0.5 * (self.__iteration_error / (len(self.samples) * 1))

    # 训练样本的Cost Function: 由于会在 _iteration_cost_function()计算迭代的cost function时去统一除 1/2。
    # 故在这里计算训练样本的cost function 时不除以 1/2。
    def _cost_function(self, error_value=0.0):
        self.__iteration_error += (error_value ** 2)

    def _net_input(self, features=[]):
        return  np.dot(features, self.weights)

    def _net_output(self, net_input=0.0):
        return self.activation.activate(net_input)

    def _start(self, iteration, completion):
        self.__iteration_times += 1
        self.__iteration_error += 0.0

        # 这里刻意反每一个步骤都写出来，一步步的代算清楚流程
        for index, features in enumerate(self.samples):
            # Forward
            target_value         = self.targets[index]
            net_input            = self._net_input(features)
            net_output           = self._net_output(net_input)

            # Backward
            error_value          = target_value - net_output
            derived_activation   = self.activation.partial(net_output)
            # Calculates cost function of the training sample.
            self._cost_function(error_value)
            # Updates all weights, the formula:
            # delta_value = -(target value - net output) * f'(net)
            # delta_weight = -learning rate * delta_value * x1 (Noted: 这里 learning rate 和 delta_value的负号会相）
            # new weights, e.g. new s1 = old w1 + delta_weight w1
            delta_value = error_value * derived_activation
            delta_weights = np.multiply(self.learning_rate * delta_value, features)
            new_weights = np.add(self.weights, delta_weights)
            self.weights = new_weights

        # Finished an iteration then adjusts conditions
        if(self.__iteration_times >= self.max_iteration) or (self._iteration_cost_function() <= self.convergence):
            if not completion is None:
                completion(self.__iteration_times, self.weights)
        else:
            if not iteration is None:
                iteration(self.__iteration_times, self.weights)
            self._start(iteration, completion)

    # One training sample: features -> one target
    def add_pattern(self, features=[],target=0):
        # If features is not an array that still working on here
        if not features:
            return
        # samples[features array]
        # targets[target value]
        self.samples.append(features)
        self.targets.append(target)

    def initialize_weights(self, weights=[]):
        if not weights:
            return
        self.weights = weights

    # 全零的初始权重
    def zero_weights(self):
        if not self.samples:
            return
        length = len(self.samples[0])
        for i in range(length):
            self.weights.append(0.0)

    def randomize_weights(self, min=0.0, max=1.0):
        # Float
        random         = np.random
        input_count    = len(self.samples[0])
        weights        = []
        for i in range(0, input_count):
            weights.append(random.uniform(min,max))
        self.initialize_weights(weights)

    # iteration and completion are callback functions
    def training(self, iteration, completion):
        self.__iteration_times = 0
        self.__iteration_error = 0.0
        self._start(iteration, completion)

    def predict(self, features=[]):
        return self._net_output(self._net_input(features))