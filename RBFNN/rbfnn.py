# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np

from activation.cost import Cost
from learning.sga import SGA
from picker.picker import Picker
from layers.hidden_layer import HiddenLayer
from layers.output_layer import OutputLayer
from nets.center_net import CenterNet
from nets.output_net import OutputNet
from sample import Sample

# 中心点选取方式
class PickMethod(Enum):
    Random     = 0
    Clustering = 1

class RBFNN:

    def __init__(self):
        self.samples         = [] # <Sample Object>
        self.iteration       = 0
        self.max_iteration   = 1
        self.tolerance_error = 0.001
        self.learning_rate   = 1.0
        self.hidden_layer    = HiddenLayer()
        self.output_layer    = OutputLayer()
        self.sga             = SGA()
        self.cost            = Cost()
        self.picker          = Picker()

    def reset(self):
        del self.samples[:]
        self.iteration = 0
        self.hidden_layer.clear_nets()
        self.output_layer.clear_nets()
        self.cost.clear()

    # sample: Sample Object
    def add_sample(self, sample):
        if not sample:
            return
        self.samples.append(sample)

    def add_samples(self, samples=[]):
        for sample in samples:
            self.add_sample(sample)

    # 对输出层神经元给定随机权重
    def randomize_weights(self, min=-0.25, max=0.25):
        hidden_count = len(self.hidden_layer.nets)
        for output_net in self.output_layer.nets:
            output_net.randomize_weights(hidden_count, min, max)

    def zero_weights(self):
        self.randomize_weights(0.0, 0.0)

    def add_center(self, center=[]):
        center_net = CenterNet(center)
        self.hidden_layer.add_net(center_net)

    # k: 要几个中心点（隐藏层神经元）
    # pick_method: 挑选中心点的方法
    def initialize_centers(self, k=1, pick_method=PickMethod.Random):
        self.picker.samples = self.samples
        picked_samples   = {
            PickMethod.Random: self.picker.shuffing,
            PickMethod.Clustering: self.picker.clustering
        }.get(pick_method)(k)
        # 把选取到的中心点都设定进孙菲菲层里
        for sample in picked_samples:
            center_net = CenterNet(sample.features)
            self.hidden_layer.add_net(center_net)

    def initialize_outputs(self):
        # 有几个输出
        outputs_count = len(self.samples[0].targets)
        for i in range(outputs_count):
            output_net = OutputNet()
            self.output_layer.add_net(output_net)

    # custom_sigmas: <Double>, 自订每个中心点的 Sigma, 1 center has 1 sigma.
    def training(self, iteration_callback=None, completion_callback=None, custom_sigmas=[]):
        self.iteration = 0
        self.cost.clear()

        # 先统一设定 3 个参数的学习速率
        # 1. 权重（weight)
        # 2. 中心点（center）
        # 3. 标准差（sigma）
        self.sga.uniform_learning_rate(self.learning_rate)

        # 设定中心点的标准差
        if len(custom_sigmas) > 0:
            self.hidden_layer.refresh_centers_sigma(custom_sigmas)
        else:
            # 如果没有自订的 Sigmas， 则会跑演算法去算初始通用的 Sigma
            self.hidden_layer.initialize_centers_sigma()

        # 开始训练
        while(self.iteration < self.max_iteration and self.cost.rmse > self.tolerance_error):
            self.iteration += 1
            for sample in self.samples:
                # Network Outputing
                center_nets   = self.hidden_layer.nets
                output_nets   = self.output_layer.nets
                hidden_outputs = self.hidden_layer.output(sample)  # Output RBF Values
                network_outputs = self.output_layer.output(sample, hidden_outputs)
                # Training Failed (做例外处理)
                if network_outputs == -1:
                    if completion_callback:
                        completion_callback(self, False)
                    return
                # 记录 Cost
                self.cost.add(network_outputs, sample.targets)
                # Updates centers and weights
                self.sga.update_centers(sample, center_nets, output_nets)
                self.sga.update_weights(center_nets, output_nets)

            # 所有训练样本（training samples）都跑完后为 1 迭代（Iteration）
            if iteration_callback:
                iteration_callback(self)

        # 完成训练
        if completion_callback:
            completion_callback(self, True)

    # features <Double>
    def predicate(self, features=[]):
        return self.output_layer.predicate(features, self.hidden_layer.nets)


