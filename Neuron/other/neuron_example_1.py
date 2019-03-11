# -*- coding: utf-8 -*-

from numpy import exp, array, random, dot

class NeuronLayer():
    # 动态生成权重
    def __init__(self,number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons))

class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # sigmoid微分
    def _sigmoid(self, x):
        return  1 / (1 + exp(-x))

    # sigmoid微分导函数
    def _sigmoid_derivative(self,x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # 前馈算法
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # dj-yj误差，Cost
            layer2_error = training_set_outputs - output_from_layer_2
            # 后馈算法
            layer2_delta = layer2_error * self._sigmoid_derivative(output_from_layer_2)

            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self._sigmoid_derivative(output_from_layer_1)

            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # 修正权重
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    # 前馈算法，
    def think(self,inputs):
        output_from_layer1 = self._sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self._sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

    # 打印权重
    def print_weights(self):
        print("  Layer 1 (4 neurons, each with 3 inputs):")
        print( self.layer1.synaptic_weights)
        print(" Layer2 (1 neuron, with 4 inputs")
        print( self.layer2.synaptic_weights)

if __name__ == "__main__":
    random.seed(1)

    layer1 = NeuronLayer(4, 3)

    layer2 = NeuronLayer(1, 4)

    neural_network = NeuralNetwork(layer1, layer2)

    print("Stage 1)Random starting synaptic weights:")
    neural_network.print_weights()

    # 输入数据
    training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    # 期望输出值
    training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

    # 训练
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print("Stage 2)New synaptic weights after training:")
    neural_network.print_weights()

    print("Stage 3)Considering a new situation [1,1,0] -> ?:")
    hidden_state, output = neural_network.think(array([1,0,1]))
    print(output)