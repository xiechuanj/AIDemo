# -*- coding: utf-8 -*-

# BPTT 的每个时间步 （timestep) 权重修正量记录在这里
class Timestep:

    def __init__(self):
        self.delta_bias              = 0.0
        self.delta_weights           = []  # <number>, 每个 Hidden Net 在 t 时刻的每一条权重变更量（BPTT 使用）
        self.recurrent_delta_weights = []  # <number>，每个 Hidden Net 在 t 时刻的每一条递归权重变更量 （BPTT 使用）

    def add_delta_weight(self, delta_weight):
        self.delta_weights.append(delta_weight)

    def add_recurrent_delta_weight(self, delta_weight):
        self.recurrent_delta_weights.append(delta_weight)

    def delta_weight(self, index=0):
        return self.delta_weights[index]

    def recurrent_delta_weight(self, index=0):
        return self.recurrent_delta_weights[index]