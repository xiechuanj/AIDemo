# -*- coding: utf-8 -*-



import numpy as np
from activation import Method
from net import Net
from layer import Layer


class MLP:

    def __init__(self):
        self.layers     = []   # All layers: last index is output layer, others index are hidden layers.
        self.random_max = 0.5  # Max random-scope of weights
        self.random_min = -0.5 # Min random-scope of weights

    # layer_settings[]:
    #     last index: output layer
    #     others index: hidden layers
    # e.g layer_settings = [5, 4, 2]
    #     index 0: hidden layer 1: has 5 nets
    #     index 1: hidden layer 2: has 4 nets
    #     index 2: output layer, has 2 nets:
    def build_network(self, layer_settings=[], activation=Method.Sigmoid):
        self.layers = []

        # 每颗神经元的Activation Function都可以自订，这边范例皆使用Sigmoid
        activations = []
        for setting in layer_settings:
            nets_activation = []
            for index in range(0, setting):
                nets_activation.append(activation)
            activations.append(nets_activation)

        # 取出每一个 Layer 要有几颗神经元（Net）的设定
        for layer_index in range(0, len(layer_settings)):
            new_layer = Layer.create(activations[layer_index])
            self.layers.append(new_layer)

        # 设定将隐藏层(Hidden Layers)， 以及输出层(Output Layer)互相连结在一起
        layer_count = len(self.layers)
        for from_layer_index in range(0, layer_count-1):
            from_layer = self.layers[from_layer_index]
            to_layer   = self.layers[from_layer_index + 1]
            # 从哪一层到哪一层，最小权重随机范围，最大权重随机范围
            from_layer.link_layer(to_layer, self.random_min, self.random_max)

    # training_samples: 训练样本
    # sample_targets  : 训练样本集里，每一个样本分别对应的目标输出值
    # max_iteration   : 最大训练迭代数
    def training(self, training_samples=[], sample_targets=[], max_iteration=1000):
        if len(training_samples) == 0:
            return

        # 设定初始 Input Layer 到 Hidden Layer (self.layers[0]) 的权重
        net_count = len(training_samples[0])
        for net in self.layers[0].nets:
            initial_weights = np.random.uniform(self.random_min, self.random_max, net_count)
            net.weights     = initial_weights

        iteration_cost = 10
        while iteration_cost >= 0.001 and max_iteration > 0:
            iteration_cost = self.iteration(training_samples, sample_targets)
            print("iteration %r", max_iteration)
            print("iteration_cost %r", iteration_cost)
            max_iteration -= 1

    def iteration(self, training_samples=[], sample_targets=[]):
        iteration_cost = 0.0

        for index, sample in enumerate(training_samples):
            target     = sample_targets[index]
            # Forward (前馈运算)
            # 从第 1 层 Hidden Layer 开始输入，而后一路走 link_layer 有授勾起来的 Next Layers.
            # 最后回传网络输出的阵列
            network_output = self.layers[0].layer_outputs(sample)
            # 计算本层训练样本的Cost
            iteration_cost += self.calculate_mse(network_output, target)
            # Backward (后馈运算)
            # backward_errors = map(lambda (a,b): a-b, zip(target, network_output))
            backward_errors = [a-b for a,b in zip(target, network_output)]
            output_layer = self.layers[-1]
            for error_index, net in enumerate(output_layer.nets):
                output_delta_value    = backward_errors[error_index] * net.differential_activition()
                backward_errors[error_index] = output_delta_value

            output_layer.update(backward_errors)
        return iteration_cost / (len(training_samples) * len(sample_targets[0]))

    def calculate_mse(self, output=[], targets=[]):
        mse = 0.0
        for index, target in enumerate(targets):
            mse += (target - output[index]) ** 2
        return mse * 0.5

    def predicate(self, features=[]):
        if not features:
            return float("inf")
        return self.layers[0].layer_outputs(features)