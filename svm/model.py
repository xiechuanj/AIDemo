# -*- coding: utf-8 -*-

import numpy as np
from sample import Sample
from kernel import Method, Kernel
from enum import Enum
from group import Group
import copy

class TrainingTypes(Enum):
    OneIterationFinished   = 0, # 一个迭代的结束
    AllConformedKKT        = 1, # 全部点皆符合KKT条件
    Failed                 = 2

class Model:

    iteration_times = 0

    def __init__(self):
        self.label           = float("inf")   # 预设是 Float 最大值， 代表这是一个标准只处理 2 分类的 SVM Model
                                               # Label 的原意思是用在多分类上，看这一个 Model 主要是用来分类哪一种【正样本】 的。
        self.samples         = []  # Sample Object, from sample.py
        self.weights         = []
        self.bias            = 0.0  # bias 只有 1 个
        self.groups          = {}  # 分到正样本（+1）或负样本（-1）群里：[target value] = group
        self.const_value     = 0.0
        self.tolerance_error = 0.0
        self.max_iteration   = 100
        self.kernel          = Kernel(Method.Linear)   # 预设使用线性分割（Linear)
        self.iteration_callback = None
        self.completion_callback = None
        self.examine_all         = False               # 是否遍历全部的点
        self._create_groups([1, -1])                    # 建立 +1， -1 这 2 个分类群，之后多分类会用到

        self.split_index         = 0
        self.iteration_update_count = 0

    # A sample <Sample Object> has a lot of features.
    def add_sample(self, sample):
        self.samples.append(copy.copy(sample))


    def append_sample(self, features=[], target_value=0.0):
        sample = Sample(features, target_value)
        sample.kernel.method = self.kernel.method
        self.add_sample(sample)

    def zero_weights(self, count=0):
        if count <= 0:
            count = len(self.samples[0].features)
        del self.weights[:]
        for i in xrange(0, count):
            self.weights.append(0.0)

    def clear_samples(self):
        del self.samples[:]

    def clear_groups(self):
        # 清空 group 里记录的 samples
        for target, group in self.groups.items():
            group.clear()

    # 从每一个 Sample 的target value 来逐一判断该点是属于哪一群
    def classify_to_group(self):
        self.clear_groups()
        # 再全部重新分类
        for sample in self.samples:
            to_group = self.groups.get(sample.target_value)
            if to_group:
                to_group.add_sample(sample)

    def classify(self, iteration_callback, completion_callback):
        self.iteration_callback = iteration_callback
        self.completion_callback = completion_callback
        self.iteration_times     = 0
        self.clear_groups()
        self._training()

    def predicate(self, features=[]):
        # Dirctly output the target value by formula : yi = (W^T * xi + b) or (W^T * xi - b)
        # 计算目标估值
        target_value  = -self.bias

        for sample_x in self.samples:
            if sample_x.alpha_value != 0:
                # SUM ai * yi * K(Xi * x)
                target_value += sample_x.alpha_value * sample_x.target_value * self.kernel.calculate(sample_x.features, features)
        return self.sgn(target_value)

    # 用于在预测输出时，将计算完的样本点目标值正规化成分类目标的 +1 / -1
    def sgn(self, value=0.0):
        return 1.0 if value >=0.0 else -1.0

    '''
    @ Private
    '''
    # 建立要分类的群
    def _create_groups(self, targets=[]):
        for target_value in targets:
            self.groups[target_value] = Group(target_value)

    def _training(self):

        self.iteration_times += 1
        waiting_samples = []
        if self.examine_all == True:
            waiting_samples = self._samples_without_kkt(self.split_index)
        else:
            waiting_samples = np.copy(self.samples).tolist()

        self._start_to_update(waiting_samples)


    def _completion(self):
        if self.completion_callback:
            self.classify_to_group()  # 分类到所属群里
            self.completion_callback(self.iteration_times, self.weights, self.bias, self.groups.values())

    def _iteration(self):
        if self.iteration_callback:
            self.iteration_callback(self.iteration_times, self.weights, self.bias)

    def _random_pick_index(self, avoid_index=0):
        max          = len(self.samples)
        random_index = 0
        # 整体样本数有2个，就直接选择另一个点来做
        if max == 2:
            random_index = (max - 1) - avoid_index
        else:
            # 整体样本有多个，就跑 Random Picking
            random_index = np.random.random_integers(0, max-1)
            if random_index == avoid_index:
                random_index = self._random_pick_index(avoid_index)
        return random_index

    def _update_parameters(self, update_alphas=[]):
        alphas_count = len(update_alphas)
        # 如果 update_alphas 为空，代表完成本次迭代训练， 但所有Samples 都还未全部符合 KKT 条件
        if  alphas_count == 0:
            return TrainingTypes.OneIterationFinished

        self._calculate_error_value()
        self.iteration_update_count += 1

        # If we still have over 2 samples can do match-update task
        if alphas_count > 1:
            match_sample     = update_alphas.pop(0)  # Romoved the sample from array
            self.split_index = self.samples.index(match_sample) +1
            max_index        = -1
            max_error_value  = -1.0
            for index, other_sample in enumerate(self.samples):
                # 找到误差距离绝对值最大的样本点
                error_distance = abs(other_sample.error_value - match_sample.error_value)
                if error_distance > max_error_value and index >= self.split_index:
                    max_error_value = error_distance
                    max_index       = index

            # If we successfully chose a sample
            if max_index >= 0:
                self.update_alpha(max_index, self.samples.index(match_sample))
                # 单纯检查是否所有数据都符合 KKT 条件了 ？ 还有不符合的就再递归跑本 function
                if self._all_conform_kkt() == False:
                    if self.examine_all == True:
                        update_alphas = self._samples_without_kkt(self.split_index)
                    # 将其它不符合 KKT 条件的点都再重新进行更新 weights & bias 运算， 直至所有点都运算完毕， 才 return 完成 1 迭代
                    return self._update_parameters(update_alphas)
                else:
                    # 更新完所有不符合 KKT 条件的点， 同时代表完成完整的 1 迭代运算就 return 完成
                    return TrainingTypes.AllConformedKKT
        else:
            # 挑 1 出来搭配，之后重新跑一次上次的运算
            # 这里有 2 个挑选的方式
            match_sample = update_alphas.pop(0)
            if self.examine_all == True:
                self.split_index = self.samples.index(match_sample) + 1
                update_alphas    = self._samples_without_kkt(self.split_index)
            match_index = self.samples.index(match_sample)
            self.update_alpha(self._random_pick_index(match_index), match_index)

            return self._update_parameters(update_alphas)
        # Default is failed.
        return TrainingTypes.Failed

    # Updating alpha and bias.
    def update_alpha(self, main_index, match_index):
        main = self.samples[main_index]
        match = self.samples[match_index]

        new_match_alpha = self._calculate_new_match_alpha(main, match)
        new_main_alpha =self._calculate_new_main_alpha(main, match, new_match_alpha)

        # Quickly updating the weights and bias by used 2 new alpha values
        # 1). calculates the delta weights, Formula:
        # delta main   = (new alpha 1 - old alpha 1) * target1 * x1
        # delta match  = (new alpha 2 - old alpha 2) * target2 * x2
        # delta weights = delta main + delta match
        main_factor  = (new_main_alpha - main.alpha_value) * main.target_value
        delta_main   = np.multiply(main.features, main_factor)

        match_factor = (new_match_alpha - match.alpha_value) * match.target_value
        delta_match  = np.multiply(match.features, match_factor)

        delta_weights = np.add(delta_main, delta_match)

        # 2). let original weights + delta weights to be new weights array, Formula:
        new_weights = np.add(self.weights, delta_weights)  # 这里 new_weights 会是 numpy.ndarray
        del self.weights[:]
        self.weights = new_weights.tolist()

        # 3). quickly updating bias via 2 samples (Main & Match), Formula:
        # W: weights, X: sample features, b: bias, T: sample target value (+1 / -1)
        # WX - b = T
        # -> -b = T - WX
        # b = WX -T
        # 故 new bias = new weights * X - (+1 or -1)
        # +1 或 -1 是看当前的 X 是被分到 +1 或者 -1 的标签（Target)
        # 这里会有 2 个 new bias, 再去按照条件做挑选 1 个出来用。
        # 以下有个更新 bias 的方法（ New, Old):

        # Linear method
        # new_main_bias = np.dot(self.weights, main.features) - main.target_value
        # new_match_bias = np.dot(self.weights, match.features) - match.target_value

        # Old method
        new_main_bias = self.bias + main.error_value + ((new_main_alpha - main.alpha_value) * main.target_value * self.kernel.calculate(main.features, main.features)) + ((new_match_alpha - match.alpha_value) * match.target_value * self.kernel.calculate(match.features, main.features))
        new_match_bias = self.bias + match.error_value + ((new_main_alpha - main.alpha_value) * main.target_value * self.kernel.calculate(main.features, match.features)) + ((new_match_alpha - match.alpha_value) * match.target_value * self.kernel.calculate(match.features, match.features))

        # 4). to choose the final bias or to get the average value of biases
        self.samples[main_index].alpha_value = new_main_alpha
        self.samples[match_index].alpha_value = new_match_alpha
        new_bias = 0.0
        if self._is_accept_alpha(new_main_alpha):
            new_bias = new_main_bias
        elif self._is_accept_alpha(new_match_alpha):
            new_bias = new_match_bias
        else:
            new_bias = (new_main_bias + new_match_bias) * 0.5

        # Update old bias
        self.bias = new_bias

    # 更新 Weights / Bias
    def _start_to_update(self, waiting_samples=[]):
        # if len(waiting_samples) == 0:
        #     self._completion()
        #     return
        # 更新参数（权重与偏权）后，再判断是否需要停止迭代或要继续下一迭代的训练
        training_result = self._update_parameters(waiting_samples)

        self.split_index = 0
        self.examine_all = True


        # 完成 1 个迭代的运算
        if training_result == TrainingTypes.OneIterationFinished:
            # 先判断迭代是否达到上限
            if self.iteration_times >= self.max_iteration:
                self._completion()
            elif self.iteration_update_count == 0:
                self._completion()
            else:
                # 继续迭代运算
                self.iteration_update_count = 0
                self._iteration()
                self._training()
        # 所有样本点都符合 KKT 条件
        elif training_result == TrainingTypes.AllConformedKKT:
            self._completion()
        else:
            # TrainingTypes.Failed
            self._completion()

    # 找出不符合 KKT 条件的样本点 （等待更新的样本点）
    def _samples_without_kkt(self, split_index=0):
        waiting_samples = []
        for sample in self.samples[split_index:]:
            is_conform_kkt = sample.is_confom_kkt(self.samples, self.bias, self.const_value)
            # 不符合 KKT 条件
            if is_conform_kkt == False and sample.alpha_value > 0 and sample.alpha_value < self.const_value:
                # 把要更新的样本记起来（不符合 KKT 的都为待更新样本）
                waiting_samples.append(sample)

        return waiting_samples

    # 是否所有样本都已符合 KKT, return BOOL
    def _all_conform_kkt(self):
        all_conform_kkt = True
        for sample in self.samples:
            all_conform_kkt = sample.is_confom_kkt(self.samples, self.bias, self.const_value)
            # 有任一样本点不符合 KKT 条件
            if all_conform_kkt == False:
                sum_x   = np.dot(self.weights, sample.features)
                kkt_value = sample.target_value * (sum_x - self.bias)
                break
        return all_conform_kkt

    # 计算每个Sample 的 Error Value
    def _calculate_error_value(self):
        errors = []
        for current_sample in self.samples:
            # Sample error
            error_value = 0.0
            # 跟其它的样本点做比较（计算误差值）
            for other_sample in self.samples:
                # kernel_value = 求当前的主要 current_sample 特征值与所有的 other_sample 特征值做内和积
                #                （包含目前 current_sample 自己对自己的内积），而后将内积值再传入至 kernel 做运算后的值。
                kernel_value       = self.kernel.calculate(current_sample.features, other_sample.features)
                other_target_value = other_sample.target_value
                other_alpha_value  = other_sample.alpha_value
                error_value        += (other_target_value * other_alpha_value * kernel_value)
            error_value += -self.bias - current_sample.target_value
            errors.append(error_value)
            # 将Error Value 存回去当前的样本点的误差值里
            current_sample.error_value = error_value

        return errors

    # 计算 New Matched Pattern Alpha Value & 判断其是否符合上下限范围
    def _calculate_new_match_alpha(self, main_sample, match_sample):
        old_main_alpha   = main_sample.alpha_value
        main_target     = main_sample.target_value
        # Update the alpha value of match-pattern in first
        # Old match alpha value + ( match target value * ( main error - match error)) / ((x1 * x1 ) + ( x2 * x2)  +  （2 * x1 * x2)
        old_match_alpha = match_sample.alpha_value
        match_target    = match_sample.target_value

        # 分子：match target * ( main error - match error) and it won't need to do fabs(error)
        numerator    = match_target * (main_sample.error_value - match_sample.error_value)
        # 分母
        denominator  = self.kernel.calculate(main_sample.features, main_sample.features) + self.kernel.calculate(match_sample.features, match_sample.features) - (2.0 * self.kernel.calculate(main_sample.features, match_sample.features))
        # New match alpha
        new_match_alpha = old_match_alpha + (numerator / denominator)

        # Checking the max-min limitation(检查上下限范围)
        min_scope = 0.0
        max_scope = 0.0
        # 相关讯号： If main target * match target = -1 (minor singal), using this formula:
        if main_target * match_target < 0.0:
            # Min scope is MAX( 0.0f, (old_match_alpha - old_main_alpha))
            min_scope = np.maximum(0.0, (old_match_alpha - old_main_alpha))
            # Max scope is MIN( const value, const value + old match alpha - old main alpha)
            max_scope = np.minimum(self.const_value, (self.const_value + old_match_alpha - old_main_alpha))
        else:
            # 同讯号
            # If main target * match target = 1 (plus singal), using this formula:
            # Min scope is MIN(0.0f, ( old main alpha + old match alpha - const value))
            min_scope = np.maximum(0.0, (old_main_alpha + old_match_alpha - self.const_value))
            max_scope = np.minimum(self.const_value, (old_match_alpha + old_main_alpha))

        # Compares max and min value of new match alpha value.
        # 如果 match 的 alpha 值在原公式制定的标准范围内，就什么都不处理，仅处理以下 2 个条件

        # 如果 match 的 alpha 值小于下限值，就变成下限值
        if new_match_alpha < min_scope:
            new_match_alpha = min_scope
        # 如果大于上限值，就变成上限值
        elif new_match_alpha > max_scope:
            new_match_alpha = max_scope

        return new_match_alpha

    # 更新 New Main Alpha Value
    def _calculate_new_main_alpha(self, main_sample, match_sample, new_match_alpha):
        # Formula: new main alpha = old main alpha + ( main target * match target * (old match alpha - new match alpha))
        return main_sample.alpha_value + (main_sample.target_value * match_sample.target_value * (match_sample.alpha_value - new_match_alpha))

    # 判断 New Alpha Value 是否在接受范围里
    def _is_accept_alpha(self, alpha_value=0.0):
        return True if(alpha_value > 0.0 and alpha_value < self.const_value) else False







