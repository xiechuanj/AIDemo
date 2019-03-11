# -*- coding: utf-8 -*-

from model import Model
from kernel import Method, Kernel
from svm import SVM, Model

import sys
sys.setrecursionlimit(10000)

svm = SVM()
svm.tolerance_error   = 0.1
svm.max_iteration     = 1000
svm.const_value       = 1.0
svm.kernel_method     = Method.Linear
svm.bias              = 0.0

# 6 笔 are 1
svm.add_sample([1, 2, 1], 1.0)
svm.add_sample([0, 2, 5], 1.0)
svm.add_sample([-1, 3, 1], 1.0)
svm.add_sample([-1, 3, 4], 1.0)
svm.add_sample([0, 5, 6], 1.0)
svm.add_sample([5, 2, -1], 1.0)

# 5 笔 are 2
svm.add_sample([2, 3, 0], 2.0)
svm.add_sample([3, 4, -2], 2.0)
svm.add_sample([4, 2, -1], 2.0)
svm.add_sample([4, 5, -3], 2.0)
svm.add_sample([5, 2, -1], 2.0)

# 4 笔 are 3
svm.add_sample([2, -2, 1], 3.0)
svm.add_sample([2, 0, 4], 3.0)
svm.add_sample([3, -1, 3], 3.0)
svm.add_sample([4, -2, 3], 3.0)

# svm.add_sample([1,0, 2.0], -1.0) # x1
# svm.add_sample([0.0, 0.0], -1.0)  # x2
# svm.add_sample([2.0, 2.0], -1.0)  # x3
# svm.add_sample([2.0, 0.0], 1.0)   # x4
# svm.add_sample([3.0, 0.0], 1.0)   # x5

svm.zero_weights()

def iteration_callback(iteration_times, weights, bias):
    print("iteration %r" % iteration_times)
    print("迭代权重 %r" % weights)
    print("迭代偏权 %r" % bias)
    pass

def completion_callback(iteration_times, weights, bias, groups):
    print("completion %r" % iteration_times)
    print("最终权重 %r" % weights)
    print("最终偏权 %r" % bias)
    # for group in groups:
    #     print("分群结果（%r）" %(group.target_value, group.samples))

svm.training(iteration_callback, completion_callback)

to = svm.predicate([-1, 3, 1])  # 1
print("预期分到 1， 结果分到 %r\n" % to)

to = svm.predicate([3, 4, -2])  # 2
print("预期分到 2， 结果分到 %r\n" % to)

to = svm.predicate([2, -2, 1]) # 3
print("预期分到 3， 结果分到 %r\n" % to)

# print(svm.predicate([1.0, 2.0]))
# print(svm.predicate([0.0, 0.0]))
# print(svm.predicate([2.0, 2.0]))
#
# print(svm.predicate([2.0, 0.0]))
# print(svm.predicate([3.0, 0.0]))