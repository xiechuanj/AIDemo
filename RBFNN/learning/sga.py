# -*- coding: utf-8 -*-

import numpy as np
from activation.activation import Activation

class SGA:

    def __init__(self, features=[]):
        self.weights_learning_rate   = 0.1   # Learning rate of weights updating.
        self.center_learning_rate    = 0.1   # Learning rate of centers updating.
        self.sigma_learning_rate     = 0.1   # Learning rate of sigma updating.
        self.activation              = Activation()

    def uniform_learning_rate(self, learning_rate=1.0):
        self.weights_learning_rate = learning_rate
        self.center_learning_rate  = learning_rate
        self.sigma_learning_rate   = learning_rate

    # 先更新中心点（中心点的特征值）与 Sigma
    # samples <Sample Object>
    # centers <CenterNet Object>
    # net_weights <Output Net Weights>
    def update_centers(self, sample=None, center_nets=[], output_nets=[]):
        if sample == None:
            return

        for center_index, center_net in enumerate(center_nets):
            # 运算原则是“更新中心点的同时，一并更新该中心点的 Sigma”
            # 计算 Output Error = ei(p) = di(p) - yi(p),即计算该 Center 要负担多少的输出误差量
            # SUM(wj(p) * ei(p)),
            # e.g. c1 error = (error1 * weight11 + error2 * weight12 + ... + errorN * weights1N) / (sigma^2)
            sum_cost = 0.0
            for output_net in output_nets:
                output_weight = output_net.weights[center_index]
                sum_cost     += output_weight * output_net.output_error  # wj(p) * ei(p)
            # learning_rate * (wj(p) * ei(p) / sigma^2(p)) * R(||x(p) - cj(p)||)
            center_delta_value =self.center_learning_rate * (sum_cost / (center_net.sigma ** 2)) * center_net.rbf_value
            # then * (x(p) - cj(p))
            diff_features = np.subtract(sample.features, center_net.features)
            delta_centers = np.multiply(diff_features, center_delta_value)
            new_centers = np.add(center_net.features, delta_centers)  # 中心点的新特征值： center features + delta centers

            # sigma error = ( error1 * weight11 + error2 * weight12 + ... + errorN * weight1N) / ( sigma * sigma * sigma)
            # distance = ||x(p) - cj(p)||^2 that means euclidean without sqrt(), the cj(p) is current old features of center (not update).
            distance = self.activation.eculidean_without_sqrt(sample.features, center_net.features)
            delta_sigma = self.sigma_learning_rate * (sum_cost / (center_net.sigma ** 3)) * center_net.rbf_value * distance
            new_sigma = center_net.sigma + delta_sigma

            # Updates sigma of center net.
            center_net.sigma = new_sigma

            # Updates features of center net.
            center_net.refresh_features(new_centers.tolist())

    # 最后再更新权重（因为旧权重会先不断的被共用于中心点和 Sigma 的更新运算上，故最后才能更新权重）
    def update_weights(self, center_nets=[], output_nets=[]):
        for output_net in output_nets:
            new_weights = []
            # 取出所有对应该 Output Net 的中心点
            for center_index, center_net in enumerate(center_nets):
                # learning_rate * error_value * rbf_value
                delta_weight = self.weights_learning_rate * output_net.output_error * center_net.rbf_value
                # center_net.rbf_value: 对应当前 Output Net 的“各个中心点的 RBF output value”
                # 取出Output Net 对应 Center 的权重
                old_weight = output_net.weights[center_index]
                new_weight = old_weight + delta_weight
                new_weights.append(new_weight)
            output_net.refresh_weights(new_weights)

