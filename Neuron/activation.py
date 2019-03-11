# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np


class Method(Enum):
    Tanh = 0 #Hyperbolic Tangent 变曲正切函数
    Sigmoid = 1  #Sigmoid S形函数

class Activation:
    method = Method.Sigmoid
    slope = 1.0

    def __init__(self):
        self.derivate = self.Derivative()

    def sigmoid(self, x=0.0, slope=1.0):
        return 1.0/(1.0 + np.exp(-slope *x ))

    def tanh(self, x=0.0, slope=1.0):
        return 2.0/(1.0 + np.exp(-slope * x)) - 1.0

    def activate(self, net_input = 0.0):
        return {
            Method.Sigmoid: self.sigmoid,
            Method.Tanh: self.tanh
        }.get(self.method)(net_input, self.slope)

    def partial(self, x=0.0, slope=1.0):
        return {
            Method.Sigmoid: self.derivate.sigmoid,
            Method.Tanh: self.derivate.tanh
        }.get(self.method)(x, self.slope)

    class Derivative:
        def sigmoid(self, x=0.0, slope=1.0):
            return slope * x * (1.0 - x)

        def tanh(self, x=0.0, slope=1.0):
            return (slope / 2.0) * (1.0 - (x ** 2))