# -*- coding: utf-8 -*-

from mlp import MLP
from activation import Method, Activation

sample_data = [[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [3.0, 0.0]]
sample_targets = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]

network = MLP()
network.build_network([2, 3, 2], Method.Sigmoid)
network.training(sample_data, sample_targets, 3000)

output = network.predicate([2.0, 2.0])

print("预测结果 %r" % output)