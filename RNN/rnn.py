# -*- coding: utf-8 -*-

import numpy as np
from activation import Activation, Method
from cost import Cost
from timestep import Timestep
from net import Net
from sample import Sample

class RNN:
    def __init__(self):
        self.tag           = self.__class__.__name__
        self.input_layer   = []      # <Sample Object>
        self.hidden_layer  = []      # <Net Object>
        self.output_layer  = []      # <Net Object>

        self.learning_rate = 1.0     # 学习速率
        self.max_iteration = 1       # 最大迭代数
        self.convergence   = 0.001   # 收敛误差
        self.activation    = Method.Sigmoid
        self.iteration     = 0       # 迭代次数
        self.cost          = Cost()  # Cost Function

        # Callbacks
        self.iteration_callback = None
        self.completion_callback = None

    def add_hidden_net(self, net):
        if not net:
            return
        # 这里的 hidden net 的 has_recurrent 必须要是 True (有递归层)
        self.hidden_layer.append(net)

    def add_output_net(self, net):
        if not net:
            return
        self.output_layer.append(net)

    # 新增 1 笔训练样本
    # 输入： 样本特征值、目标值
    def add_sample(self, features=[], targets=[]):
        if not features or not targets:
            return
        sample = Sample(features, targets)
        self.input_layer.append(sample)

    # 新增多笔训练样本
    # 输入： 多笔样本的特征值、目标值
    def add_samples(self, sample_features=[], sample_targets=[]):
        for index, features in enumerate(sample_features):
            self.add_sample(features, sample_targets[index])

    def randomize_weights(self, min=0.0, max=1.0):
        sample       = self.input_layer[0]
        input_count  = len(sample.features)
        hidden_count = len(self.hidden_layer)
        for hidden_net in self.hidden_layer:
            hidden_net.randomize_weights(input_count, min, max)
            hidden_net.randomize_recurrent_weights(hidden_count, min, max) # Recurrent Layer 的神经元数量是跟 Hidden Layer 是同步的

        for output_net in self.output_layer:
            output_net.randomize_weights(hidden_count, min, max)

    def uniform_activation(self, activation=Method.Sigmoid):
        self.activation = activation
        for hidden_net in self.hidden_layer:
            hidden_net.activation_method = activation
        for output_net in self.output_layer:
            output_net.activation_method = activation

    def create_nets(self, net_count, to_layer, has_recurrent):
        for i in range(net_count):
            net = Net(has_recurrent)
            to_layer.append(net)

    def create_hidden_layer_nets(self, net_count=1):
        self.create_nets(net_count, self.hidden_layer, True)

    def create_output_layer_nets(self, net_count=1):
        self.create_nets(net_count, self.output_layer, False)

    # sample_features: <List<Double>>, 包了多个预测样本的特征阵列
    def predicate(self, sample_features=[], completion_callback=None):
        for features in sample_features:
            network_outpus = self._forward(features)
            if completion_callback:
                completion_callback(features, network_outpus)

    # 是否继续训练 Network
    def go_ahead(self):
        under_max_iteration = self.iteration < self.max_iteration # 未达到最大迭代数
        over_cost_goal      = self.cost.mse > self.convergence # 未达到收敛误差
        if self.iteration == 0: # 是一开始
            over_cost_goal = True
        return under_max_iteration and over_cost_goal

    def training(self, iteration_callback, completion_callback):
        self.iteration_callback = iteration_callback
        self.completion_callback = completion_callback
        total                    = len(self.input_layer)
        # 未达到最大迭代数 && Cost Function 还未达到收敛误差
        while(self.go_ahead()):
            self.cost.remove()
            self.iteration += 1
            for index, sample in enumerate(self.input_layer):
                self._training_network(sample)
                current_size = index + 1
                # 是最后一笔 （Full-BPTT)
                if current_size == total:
                    self._bptt_update()

            # 已跑完 1 迭代
            if  self.iteration_callback:
                self.iteration_callback(self.iteration, self)

        # 训练结束（达到最大迭代数或已达到收敛误差）
        if self.completion_callback:
            self.completion_callback(self.iteration, self)

    '''
    @ Private
    '''
    def _bptt_update(self):
        for hidden_net in self.hidden_layer:
            hidden_net.renew()

        for output_net in self.output_layer:
            output_net.renew()

    def _forward(self, features=[]):
        # 先取出 Recurrent Layer　神经元的输出值
        recurrent_outputs =[]
        for hidden_net in self.hidden_layer:
            recurrent_outputs.append(hidden_net.output_value)

        # Forward Pass
        hidden_outputs = []
        for hidden_net in self.hidden_layer:
            net_output = hidden_net.net_output(features, recurrent_outputs)
            hidden_outputs.append(net_output)

        # Network Output
        network_outputs = []
        for output_net in self.output_layer:
            net_output = output_net.net_output(hidden_outputs)
            network_outputs.append(net_output)

        return network_outputs

    # sample: Sample Object
    def _training_network(self, sample):
        if not sample:
            return
        # Forward Pass
        network_outputs = self._forward(sample.features)

        self.cost.add(network_outputs, sample.targets)

        # Backward Pass
        # To calculate the deltas of output-layer.
        for output_index, output_net in enumerate(self.output_layer):
            # Calculating deltas of output nets.
            target_value       = sample.targets[output_index]
            output_value       = network_outputs[output_index]
            output_error       = target_value - output_value
            output_net.delta_value = (output_error * output_net.output_partial)

        # To calculate the deltas of output-layer to hidden-layer, and record the outputs of nets to an array.
        hidden_outputs = []
        recurrent_outputs = []
        # 开始计算 Hidden Layer 的权重修正量(delta)
        for hidden_index, hidden_net in enumerate(self.hidden_layer):
            # 先算从 Output Layer 到 Hidden Layer 的每颗 Hidden Net 的误差量: SUM(delta[t][k] * w[t][hk])
            # Output Layer to Hidden Layer.
            sum_delta = 0.0
            for output_net in self.output_layer:
                sum_delta += output_net.delta_value * output_net.weight_for_index(hidden_index)

            # Recurrent-layer to hidden-layer.
            sum_recurrent_delta = 0.0
            for recurrent_net in self.hidden_layer: # 这里 recurrent_net 跟 hidden_net 是同一颗,只是取不同名便于区别计算
                sum_recurrent_delta += recurrent_net.delta_value * recurrent_net.recurrent_weight_for_index(hidden_index)

            # Hidden nets delta: (SUM(output-net-delta[t][k] * w(jk)) + SUM(recurrent-net-delta[t+1][h] * w(h'h))) * f'(hidden-net-output)
            hidden_net.delta_value = (sum_delta + sum_recurrent_delta) * hidden_net.output_partial

            # To record the net output for backpropagation updating.
            hidden_outputs.append(hidden_net.output_value)
            recurrent_outputs.append(hidden_net.previous_output)

        # 计算输出层到输入层每一条的权重变化量(delta weights), 并记录至每一个神经元(net)里的 timesteps
        # To update weights of net-by-net between output-layer, hidden-layer and input-layer.
        for output_net in self.output_layer:
            output_net.calculate_delta_weights(self.learning_rate, hidden_outputs)

        # 取出输入的特征值来计算输入层到输出层的权重变化量
        # To fetch the outputs of input-layer for updating weights of hidden-layer.
        inputs = sample.features
        for hidden_net in self.hidden_layer:
            hidden_net.calculate_delta_weights(self.learning_rate, inputs, recurrent_outputs)