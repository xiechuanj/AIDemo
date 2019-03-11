# -*- coding: utf-8 -*-

class Group:

    def __init__(self, target_value=0):
        self.target_value = target_value    # 本群的目标值
        self.samples      = []              # 分到本群的样本点

    def add_sample(self, sample=[]):
        if not sample:
            return
        self.samples.append(sample)

    def clear(self):
        del self.samples[:]