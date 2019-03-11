# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np

class Method(Enum):
    Sigmoid   = 0 # Sigmoid S形函数
    Tanh      = 1 # Hyperbolic Tangent 双曲正切函数
    SGN       = 2 # 硬极限
    RBF       = 3 # 径向基（高斯）
    ReLU      = 4 # 修正线性单元（线性整流函数）
    LeakyReLU = 5 # 带泄露修正线性单元
    ELU       = 6 # 指数线性单元

class Activation(object):

    def __init__(self):
        self.derivative   = self.Derivative()
        self.slope        = 1.0
        self.sigma        = 2.0
        self.method       = Method.Sigmoid
        self.nonlinears   = [Method.Sigmoid, Method.Tanh, Method.RBF]  # 非线性函式

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-self.slope * x))

    def tanh(self, x):
        return 2.0 / (1.0 + np.exp(-self.slope * x)) - 1.0

    def sgn(self, x):
        return 1.0 if x >= 0.0 else -1.0

    def rbf(self, x):
        return np.exp(-x / (2.0 * (self.sigma ** 2)))

    def reLU(self, x):
        return np.maximum(0.0, x)

    def leakyReLU(self, x):
        return np.maximum(0.01 * x, x)

    def eLU(self, x):
        return x if x >= 0.0 else 0.01 * (np.exp(x) - 1.0)

    def activate(self, net_input=0.0):
        return {
            Method.Sigmoid   : self.sigmoid,
            Method.Tanh      : self.tanh,
            Method.SGN       : self.sgn,
            Method.RBF       : self.rbf,
            Method.ReLU      : self.reLU,
            Method.LeakyReLU : self.leakyReLU,
            Method.ELU       : self.eLU
        }.get(self.method)(net_input)

    def partial(self, x=0.0):
        return {
            Method.Sigmoid   : self.derivative.sigmoid,
            Method.Tanh      : self.derivative.tanh,
            Method.SGN       : self.derivative.sgn,
            Method.RBF       : self.derivative.rbf,
            Method.ReLU      : self.derivative.reLU,
            Method.LeakyReLU : self.derivative.leakyReLU,
            Method.ELU       : self.derivative.eLU
        }.get(self.method)(x)

    @property
    # 判断活化函式是否为线性
    def is_linear(self):
        return False if self.method in self.nonlinears else True

    @property
    # slope 越大， 梯度越大， 线越窄；反之，梯度越小，线越平缓
    def slope(self):
        return self.derivative.slope

    @property
    def sigma(self):
        return self.derivative.sigma

    @slope.setter
    def slope(self, value=1.0):
        self.derivative.slope = value

    @sigma.setter
    def sigma(self, value=2.0):
        self.derivative.sigma = value

    class Derivative:

        slope = 1.0
        sigma = 2.0

        def sigmoid(self, x):
            return self.slope * x * (1.0 - x)

        def tanh(self, x):
            return (self.slope / 2.0) * (1.0 - (x ** 2))

        def rbf(self, x):
            return -((2.0 * x) / (2.0 * (self.sigma ** 2))) * np.exp(-x / (2.0 * (self.sigma ** 2)))

        def sgn(self, x):
            return 1.0 if x >= 0.0 else -1.0

        def reLU(self, x):
            return 1.0 if x > 0.0 else 0.0

        def leakyReLU(self, x):
            return 1.0 if x > 0.0 else 0.01

        def eLU(self, x):
            return x if x >= 0.0 else 0.01 * np.exp(x)