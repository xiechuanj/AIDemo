# -*- coding: utf-8 -*-

from neuron import Neuron
from activation import Method

'''
Example
'''
neuron = Neuron()
neuron.max_iteration       = 200            # 最大迭代数
neuron.convergence         = 0.001          # 收敛误差
neuron.activation.method   = Method.Sigmoid # 活化函式
neuron.activation.slope    = 1.0            # Sigmoid的坡度（斜率）


# 设定训练样本
neuron.add_pattern([1.0, -2.0, 0.0, -1.0], 0.0)
neuron.add_pattern([0.0, 1.5, -0.5, -1.0], 1.0)

# 3 种权重设定方式
# 1）. 自设
neuron.initialize_weights([1.0, -1.0, 0.0, 0.5])
# 2). 随机范围
# neuron.randomize_weights(0.0, 1.0)
# 3). 全零
# neuron.zero_weights()

# 回呼函式： 迭代训练时触发
def iteration_callback(iteration=0, weights=[]):
    print ("Doing %r iteration: %r" % (iteration, weights))
# 回呼函式:  训练结束时触发
def completion_callback(total_iteration=0, weights=[]):
    print ("Done %r iteration: %r" % (total_iteration, weights))
    # 预测
    predicted_value = neuron.predict([1.0, -2.0, 0.0, -1.0])
    print ("Predicted %r" % predicted_value)

# 开始训练
neuron.training(iteration_callback, completion_callback)
