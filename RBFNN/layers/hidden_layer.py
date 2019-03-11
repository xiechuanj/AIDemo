# -*- coding: utf-8 -*-

from activation.activation import Activation
import numpy as np

class HiddenLayer:

    nets   = [] # <CenterNet Object>, a net is a center.
    activation = Activation()

    def clear_nets(self):
        del self.nets[:]

    def add_net(self, net=None):
        if net == None:
            return
        self.nets.append(net)

    def add_nets(self, nets=[]):
        if not nets:
            return
        self.clear_nets()
        self.nets = nets

    # Those centers of network must be set up their sigmas (fixed common sigma or 1 center has 1 sigmas).
    # 设定所有 Centers 初始 Sigma
    def initialize_centers_sigma(self):
        # 先取出所有中心点里离彼此最远的距离
        max_distance = -1.0
        for center_net in self.nets:
            for another_center_net in self.nets:
                distance = self.activation.eculidean(center_net.features, another_center_net.features)
                if distance > max_distance:
                    max_distance = distance
        # 再算标准差
        nets_count  = len(self.nets)  # 有几个中心点
        common_sigma = (max_distance / np.sqrt(nets_count)) if max_distance > 0.0 else 1.0
        # 设定每个神经元的标准差
        for center_net in self.nets:
            center_net.sigma = common_sigma

    # 自订所有 Centers 各自专属的 Sigma
    def refresh_centers_sigma(self, sigmas=[]):
        for index, center_net in enumerate(self.nets):
            center_net.sigma = sigmas[index]

    # Hidden Layer Outputs (RBF Values)
    def output(self, sample=None):
        if sample == None:
            return  -1
        rbf_values = []
        for center_net in self.nets:
            # RBF activation function
            rbf_value = self.activation.activate(sample.features, center_net.features, center_net.sigma)
            rbf_values.append(rbf_value)
            # To record this RBF value back to center_net
            center_net.rbf_value = rbf_value
        return rbf_values
