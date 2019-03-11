# -*- coding: utf-8 -*-

import copy
import numpy as np
from activation import Method, Activation

class Net:

    def __init__(self, method=Method.Sigmoid):
        self.weights            = []  # Current weights
        self.old_weights        = []  # Last time weights
        self.output             = 0.0 # Neuron output
        self.inputted_features  = []  # Inputted features
        self.summed_signal      = 0.0 # Summed singal (the summation of input)
        self.learning_rate      = 0.8 # Learning rate
        self.activition         = Activation()  # Activation function

    # 加总信号
    def summarize_inputs(self, features=[]):
        if not features:
            return 0.0
        self.inputted_features = copy.deepcopy(features)
        self.summed_signal     = np.dot(self.inputted_features, self.weights)  # a = X * W
        self.output            = self.activition.activate(self.summed_signal)  # b = f(a)
        return self.output

    # 更新权重
    def update_weights(self, error_value=0.0):
        self.old_weights = copy.deepcopy(self.weights)
        for index, old_weight in enumerate(self.old_weights):
            # new weight          = old weight + learning rate * error_value(i) * input
            new_weight            = old_weight + self.learning_rate * error_value * self.inputted_features[index]
            self.weights[index]   = new_weight

    def differential_activition(self):
        return self.activition.differentiate(self.output)