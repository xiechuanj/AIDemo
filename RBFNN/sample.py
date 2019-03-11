# -*- coding: utf-8 -*-

import copy

class Sample:

    def __init__(self, features=[], targets=[]):
        self.features = self._copy(features)
        self.targets  = self._copy(targets)

    def _copy(self, array=[]):
        return copy.deepcopy(array) if array else []

    def add_features(self, features=[]):
        self.features = self._copy(features)

    def add_targets(self, targets=[]):
        self.targets = self._copy(targets)