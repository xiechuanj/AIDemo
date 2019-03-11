# -*- coding: utf-8 -*-

import copy

class Sample:

    def __init__(self, features=[], targets=[]):
        self.features = copy.deepcopy(features)
        self.targets = copy.deepcopy(targets)