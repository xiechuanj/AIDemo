# -*- coding: utf-8 -*-

from rnn import RNN
from activation import Method

'''
Samples: 数字辨识演示（0 ~ 9）
'''
# 0 ~ 9 （the numbers)
features = [# 0
            [1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 0, 0, 0, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 0, 0, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1],

            # 1
            [0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1],

            # 2
            [1, 0, 0, 0, 1, 1, 1, 1, 1,
            1, 0, 0, 0, 1, 0, 0, 0, 1,
            1, 0, 0, 0, 1, 0, 0, 0, 1,
            1, 1, 1, 1, 1, 0, 0, 0, 1],

            # 3
            [1, 0, 0, 0, 1, 0, 0, 0, 1,
            1, 0, 0, 0, 1, 0, 0, 0, 1,
            1, 0, 0, 0, 1, 0, 0, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1],

            # 4
            [1, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1],

            # 5
            [1, 1, 1, 1, 1, 0, 0, 0, 1,
            1, 0, 0, 0, 1, 0, 0, 0, 1,
            1, 0, 0, 0, 1, 0, 0, 0, 1,
            1, 0, 0, 0, 1, 1, 1, 1, 1],

            # 6
            [1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 0, 0, 0, 1, 0, 0, 0, 1,
            1, 0, 0, 0, 1, 0, 0, 0, 1,
            1, 0, 0, 0, 1, 1, 1, 1, 1],

            # 7
            [1, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1],

            # 8
            [1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 0, 0, 0, 1, 0, 0, 0, 1,
            1, 0, 0, 0, 1, 0, 0, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1],

            # 9
            [1, 1, 1, 1, 1, 0, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1]]

targets = [ # 0
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # 1
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            # 2
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            # 3
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            # 4
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            # 5
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            # 6
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            # 7
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            # 8
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            # 9
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

rnn = RNN()
rnn.max_iteration = 500
rnn.convergence  = 0.001
rnn.learning_rate = 0.5

rnn.add_samples(features, targets)
rnn.create_hidden_layer_nets(40)
rnn.create_output_layer_nets(10)

rnn.randomize_weights(-0.25, 0.25)
rnn.uniform_activation(Method.Sigmoid)

# 回呼函式： 迭代训练时触发
def iteration_callback(iteration, network):
    print("Iteration %r cost %r" % (iteration, network.cost.mse))

def predicated_callback(features, network_outputs):
    print("Predicated %r" % network_outputs)

# 回呼函式： 训练结果时触发
def completion_callback(total_iteration, network):
    print("Done %r" % total_iteration)
    network.predicate(features, predicated_callback)

rnn.training(iteration_callback, completion_callback)