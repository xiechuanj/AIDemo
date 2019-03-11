# -*- coding: utf-8 -*-

import numpy as np
from model import Model
from group import Group
from kernel import Method, Kernel
from sample import Sample
import copy

# 多分类 SVM (Multi-Classfication SVM)，算法： 1 对多
class SVM:

    def __init__(self):
        self.samples         = {}                 # [Label (target value)] = [Sample]
        self.kernel_method   = Method.Linear
        self.models          = []                 # <Model>
        self.tolerance_error = 0.001
        self.max_iteration   = 200
        self.const_value     = 1.0
        self.bias            = 0.0
        self.weights         = []                 # 初始权重

    # 初始权重
    def initialize_weights(self, weights=[]):
        if not weights:
            return
        self.weights = np.copy(weights).tolist()

    # 全零的初始权重
    def zero_weights(self):
        all_samples    = list(self.samples.values())    # 因为 self.samples 是 {key: []}, 所以取出来的 values 都会再被 锯一个list里
        label_samples  = all_samples[0]
        sample         = label_samples[0]         # 要这样取出 sample object
        features       = sample.features
        dimension      = len(features)
        weights        = []
        for i in range(dimension):
            weights.append(0.0)
        self.initialize_weights(weights)

    def add_sample(self, features=[], target_value=0.0):
        if not features:
            return
        # 取出指定的 label (target value) samples 的 list (array)
        label_samples = self.samples.get(target_value)
        if not label_samples:
            label_samples = []
            self.samples[target_value] = label_samples
        # 建立 Sample 物件
        sample     = Sample(features, target_value)
        sample.kernel.method = self.kernel_method
        sample.index = len(label_samples)
        label_samples.append(sample)   # Reference memory

    def training(self, iteration_callback, completion_callback):
        all_labels = self.samples.keys()
        if not all_labels:
            return
        # 是标准只分 2 类的模型，只需要 1 个 model ( +1 / -1)
        if self._is_standard_binary(all_labels):
            model       = self._create_model()
            model.label = "STANDARD"
            self.models.append(model)
            self._reset_model_samples(1, model)
            model.classify(iteration_callback, completion_callback)   # callbacks 后面再补上， 外面 SVM 这里要再包一层自己的 callback, 才能让外部知道
        else:
            # 多分类，需要多个 models (例外状况： 只分 1 类的情况， 也会跑进来这里做单独运算）
            for label in all_labels:
                model       = self._create_model()
                model.label = label
                self.models.append(model)
                self._reset_model_samples(label, model)
                model.classify(iteration_callback, completion_callback)

    def predicate(self, features=[]):
        # 设计方法
        # 预测值看被分到谁是 +1 属于该， 这里也是要判断是只有一个 model 在做 2 分类（+1 / -1）， 这是有多个 models 在做分类，
        # 回传的值会不同，如果只是单纯 1 个 model 做 2 分类，哪回传值是 +1 / -1,
        # 而如果是多分类，那回传值则是 model.label(被分到哪个 label)
        # 而如果多分类里，预测结果有 2 个以上都为 +1 （都归于该群，代表该点在这几个群的交界处），那就随机挑选 1 个群来当分类结果即可。

        # 是单纯 2 分类： 只有 1 个 SVM Model
        if len(self.models) == 1:
            # 直接回传被分类到 +1 或 -1
            model = self.models[0]
            return model.predicate(features)
        else:
            # 多分类： 多个 SVM Model
            classified_to_labels = [] # 记录被分类到哪些群（正样本）
            for model in self.models:
                predicated_value = model.predicate(features)
                if predicated_value == 1:
                    classified_to_labels.append(model.label)
                print("model %r -> %r" % (model.label, predicated_value))
            # 回传被分类到哪一个 Label
            count = len(classified_to_labels)
            if count == 0:
                return "Failed Classification"
            elif count == 1:
                return classified_to_labels[0]
            else:
                return np.random.choice(classified_to_labels)

    # 是否为标准 +1 / -1 二分类
    def _is_standard_binary(self, all_lables=[]):
        # Labels 有 2 个 and 和 -1 都包含在里面
        return True if(len(all_lables) == 2) and (1 in all_lables) and (-1 in all_lables) else False

    # 设定 model 要训练的样本集
    # positive_label: 谁是正样本（classify target)
    def _reset_model_samples(self, positive_label=1.0, model=None):
        model.clear_samples()
        for label, samples in self.samples.items():
            for sample in samples:
                # 设定成正样本数据
                if label == positive_label:
                    sample.target_value = 1
                else:
                    # 其它的就都是负样本
                    sample.target_value = -1
                model.add_sample(sample)

    # 建立一个初始化的 SVM Model
    def _create_model(self):
        model                 = Model()
        model.tolerance_error = self.tolerance_error
        model.max_iteration   = self.max_iteration
        model.const_value     = self.const_value
        model.bias            = self.bias
        model.weights         = copy.deepcopy(self.weights)
        model.kernel.method   = self.kernel_method
        return model