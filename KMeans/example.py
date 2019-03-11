# -*- coding: utf-8 -*-

from kmeans import KMeans
from center import Choice

# 17笔
samples = [
    [5, 4], [3, 4], [2, 5], [9, 8], [3, 20],
    [1, 1], [1, 2], [2, 2], [3, 2], [3, 1],
    [6, 4], [7, 6], [5, 6], [6, 5], [7, 8],
    [3, 12], [5, 20]
]

kmeans = KMeans()

for features in samples:
    kmeans.add_sample(features)

kmeans.center_choice = Choice.Plus       # K-Means++ 算法
kmeans.make_centers(k=3)                 # 目标分 3 群
kmeans.max_iteration = 100               # 最大训练 100 迭代
kmeans.convergence   = 0.001             # 收敛误差
kmeans.setup()                           # 在开始训练前跑个设定

# 每迭代的回呼函式
def iteration_callback(iteration_times, groups):
    print ("iteration %r" % iteration_times)
    for group in groups:
        print("新旧群心 （%r） : % r -> %r" % (group.tag, group.old_center,group.center))

# 完成训练时的回呼函式
def completion_callback(iteration_times, groups, sse):
    print ("completion %r and sse %r" % (iteration_times, sse))
    for group in groups:
        print ("最终的群心 （%r）: %r, %r" % (group.tag, group.center, group.samples))

# 开始训练
kmeans.run(iteration_callback, completion_callback)

# 对新样本进行分类后的结果回呼函式
def classified_callback(point, group):
    print("%r classified to %r" % (point, group.tag))

# 用已训练好的模型来对新样本进行分类
kmeans.classify([[3, 4], [5, 6]], classified_callback)