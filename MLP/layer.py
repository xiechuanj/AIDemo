# -*- coding: utf-8 -*-

import copy
import numpy as np
from activation import Method
from net import Net

class Layer:
    def __init__(self):
        self.nets           = []
        self.previous_layer = None
        self.next_layer     = None
        self.inputs         = []
        self.outputs        = []

    # Layer.create()
    @classmethod
    def create(self, activation_methods=[]):
        layer   = Layer()
        layer.nets  = []
        for method in activation_methods:
            net = Net(method)
            layer.nets.append(net)
        return layer

    # next_layer: 要连接的下一层
    def link_layer(self, next_layer=None, min_weight=-0.5, max_weight=0.5):
        if next_layer == None:
            return

        self.next_layer  = next_layer
        next_layer.previous_layer = self   # 自己是下一层的上一层


        # 设定下一层的初始权重
        net_cout = len(self.nets)
        for net in next_layer.nets:
            # 本层有几颗神经元（net_count)，则下一层的神经元就会几条权重(initial_weights)
            initial_weights = np.random.uniform(min_weight, max_weight, net_cout)
            net.weights     = initial_weights

    # Layer outputs
    def layer_outputs(self, features=[]):
        self.inputs = features
        self.outputs = []
        for net in self.nets:
            net_output = net.summarize_inputs(self.inputs)
            self.outputs.append(net_output)

        # 本层的输出 = 下一层的输入（features)
        return self.next_layer.layer_outputs(self.outputs) if self.next_layer != None else self.outputs

    # error_value: 上一层每一颗的 error value (delta value: 误差修正量)
    # e.g. error_values = [net1_error, net2_error, net3_error]
    def update(self, error_values=[]):
        # 更新此 Layer 所属神经元的权重
        for update_index, net in enumerate(self.nets):
            net.update_weights(error_values[update_index])

        if self.previous_layer != None:
            # 处理此 Layer net的Error变成每一颗net的Error Value后，丢到上一层更新
            update_values = []
            for index, net in enumerate(self.nets):
                # 依照每条权重来分配误差修正量（每一颗神经元所总共要负担的量)
                # apportion_errors = map(lambda w: w * error_values[index], net.old_weights)
                apportion_errors = [w * error_values[index] for w in net.old_weights]
                if index == 0:
                    update_values = apportion_errors
                else:
                    # update_values = map(lambda (a, b): a + b, zip(update_values,apportion_errors))
                    update_values = [a + b for a,b in zip(update_values,apportion_errors)]

            # 计算上一层每颗神经元的总误差的正量
            for error_index in range(0, len(self.previous_layer.nets)):
                previous_net = self.previous_layer.nets[error_index]
                total_error = update_values[error_index] * previous_net.differential_activition()
                update_values[error_index] = total_error

            self.previous_layer.update(update_values)
        else:
            return True



