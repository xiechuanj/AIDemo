# -*- coding: utf-8 -*-

from activation.activation import Activation

class OutputLayer:

    nets    = [] # <OutputNet Object>
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

    # Doing Network Output
    # sample <Sample Object>
    # rbf_values <Double>
    def output(self, sample=None, rbf_values=[]):
        if sample == None:
            return -1
        # Forward Pass
        network_outputs = []
        # rbf_values are Centers outputs to Output Layer nets, the output1, output2, ..., outputN.
        for output_index, output_net in enumerate(self.nets):
            output_value   = output_net.output(rbf_values)
            target_value   = sample.targets[output_index]
            output_net.output_error  = target_value - output_value
            network_outputs.append(output_value)
        return network_outputs

    def predicate(self, sample_features=[], centers=[]):
        if not sample_features:
            return []

        network_outputs = []
        rbf_values      = []
        # 单纯 forward(前馈）不计算误差
        for center_net in centers:
            rbf_value = self.activation.activate(sample_features, center_net.features, center_net.sigma)
            rbf_values.append(rbf_value)

        for output_net in self.nets:
            network_outputs.append(output_net.output(rbf_values))

        return network_outputs