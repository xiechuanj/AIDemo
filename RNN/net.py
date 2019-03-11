# -*- coding: utf-8 -*-

import numpy as np
from activation import Activation, Method
from net_output import NetOutput
from timestep import Timestep

class Net(object):

    # Read, Write
    activation_method    = None

    # Readonly
    output_value            = None # 输出值
    output_partial          = None
    previous_output         = None
    previous_output_partial = None

    def __init__(self, has_recurrent=False):
        self.weights           = [] # <number>
        self.recurrent_weights = [] # <number>
        self.bias              = 0.0
        self.delta_value       = 0.0 # Current delta value will be next delta value.
        self.has_recurrent     = has_recurrent # Has recurrent inputs ? (hidden net has recurrent, but output net not.
        self.activation        = Activation()  # 活化函式的 Get, Set 都在这里： net.activation.method.
                                               # 有另外开 self.activation_method 来方便存取
        self.output            = NetOutput()
        self.timesteps         = []  # <Timestep Object>

    # weights<number>
    def reset_weights(self, weights=[]):
        if not weights:
            return
        self.weights = np.copy(weights).tolist()

    # recurrent_weights<number>
    def reset_recurrent_weights(self, recurrent_weights=[]):
        if not recurrent_weights:
            return
        self.recurrent_weights = np.copy(recurrent_weights).tolist()

    def weight_for_index(self, index=0):
        return self.weights[index]

    def recurrent_weight_for_index(self, index=0):
        return self.recurrent_weights[index]

    # bias 也在这里归零
    def remove_all_weights(self):
        del self.weights[:]
        del self.recurrent_weights[:]
        self.bias  = 0.0

    # Randomizing weights and bias.
    def randomize_weights(self, random_count=1, min=-0.5, max=0.5):
        del self.weights[:]
        self.bias = 0.0
        random = np.random
        for i in range(0, random_count):
            self.weights.append(random.uniform(min, max))

        self.bias = random.uniform(min, max)

    def randomize_recurrent_weights(self, random_count=1, min=-0.5, max=0.5):
        if self.has_recurrent:
            del self.recurrent_weights[:]
            random = np.random
            for i in range(0, random_count):
                self.recurrent_weights.append(random.uniform(min, max))

    # Net output. (Hidden net with recurrent, Output net without recurrent)
    def net_output(self, inputs=[], recurrent_outputs=[]):
        # 先走一般前馈（Forward）至 Hidden Layer
        summed_singal = np.dot(inputs, self.weights) + self.bias
        # 如果有递归层，再走递归（Recurrent) 至 Hidden Layer
        if len(recurrent_outputs) > 0:
            summed_singal += np.dot(recurrent_outputs, self.recurrent_weights)

        # 神经元输出
        output_value = self.activation.activate(summed_singal)
        self.output.add_sum_input(summed_singal)
        self.output.add_output_value(output_value)
        return output_value

    def clear(self):
        self.output.refresh()

    # For hidden layer nets to calculate their delta weights with recurrent layer,
    # and for output layer nets to calculate their delta weights without recurrent layer.
    # layer_outputs: hidden layer outputs or output layer outputs.
    def calculate_delta_weights(self, learning_rate=1.0, layer_outputs=[], recurrent_outputs=[]):
        # 利用 Timestep 来当每一个 BP timestep 算权重修正值时的记录容器
        timestep = Timestep()
        # For delta bias.
        timestep.delta_bias = learning_rate * self.delta_value
        # For delta of weights.
        for weight_index, weight in enumerate(self.weights):
            # To calculate and delta of weight.
            last_layer_output = layer_outputs[weight_index]
            # SGD: new w = old w + (-learning rate * delta_value * x)
            #      -> x 可为 b[t][h] (hidden output) 或 b[t-1][h] (recurrent output) 或 x[i] (input feature)
            # Output layer 的 delta_value = aE/aw[hk] = -error value * f'(net)
            # Hidden layer 的 delta_value = aE/aw[ih] = SUM(delta_value[t][hk] * w[hk] + SUM(delta_value[t+1][h'h] * w
            delta_weight = learning_rate * self.delta_value * last_layer_output;
            timestep.add_delta_weight(delta_weight)

        # For delta of recurrent weights. (Noted: Output Layer is without Recurrent)
        for recurrent_index, recurrent_weight in enumerate(self.recurrent_weights):
            last_recurrent_output = recurrent_outputs[recurrent_index]
            recurrent_delta_weight = learning_rate * self.delta_value * last_recurrent_output
            timestep.add_recurrent_delta_weight(recurrent_delta_weight)

        self.timesteps.append(timestep)

    def renew_weights(self, new_weights=[], new_recurrent_weights=[]):
        if not new_weights and not new_recurrent_weights:
            return
        self.remove_all_weights()
        self.reset_weights(new_weights)
        self.reset_recurrent_weights(new_recurrent_weights)

    def renew_bias(self):
        sum_changes = 0.0
        for timestep in self.timesteps:
            sum_changes += timestep.delta_bias
        # new b(j) = old b(j) + [-L * -delta(j)]
        self.bias += sum_changes

    # Renew weights and bias.
    def renew(self):
        # 累加每个 Timestep 里相同 Index 的 Delta Weight
        new_weights       = []
        new_recurrent_weights  = []

        for weight_index, weight in enumerate(self.weights):
            # For normal weight.
            sum_delta_changes = 0.0
            for timestep in self.timesteps:
                sum_delta_changes += timestep.delta_weight(weight_index)
            new_weight = weight + sum_delta_changes
            new_weights.append(new_weight)

        for recurrent_index, recurrent_weight in enumerate(self.recurrent_weights):
            # for recurrent weight.
            sum_recurrent_changes = 0.0
            for timestep in self.timesteps:
                sum_recurrent_changes += timestep.recurrent_delta_weight(recurrent_index)
            new_recurrent_weight = recurrent_weight + sum_recurrent_changes
            new_recurrent_weights.append(new_recurrent_weight)

        self.renew_weights(new_weights, new_recurrent_weights)
        self.renew_bias()

        del self.timesteps[:]
        self.delta_value = 0.0 # 一定要归零， 因为走 BPTT 的原故

    '''
    @ Getters that all Readonly
    '''
    @property
    # The last output value from output_values.
    def output_value(self):
        return self.output.last_output_value

    @property
    # The last output value partial from output_values.
    def output_partial(self):
        # 是线性输出的活化函式（e.g. SGN, ReLU ... etc.），必须用输入信号（Sum Input）来求函式偏微分
        # 非线性的活化函式则用输出值(Output Value)来求偏微。
        output_value = self.output.last_sum_input if self.activation.is_linear == True else self.output_value
        return self.activation.partial(output_value)

    @property
    # The last moment output. e.g. b[t-1][h]
    def previous_output(self):
        return self.output.previous_output

    @property
    # 上一刻的输出偏微分
    def previous_output_partial(self):
        output_value = self.output.previous_sum_input if self.activation.is_linear == True else self.previous_output
        return self.activation.partial(output_value)

    @property
    def activation_method(self):
        return self.activation.method

    '''
    @ Setter
    '''
    @activation_method.setter
    def activation_method(self, method):
        self.activation.method = method
