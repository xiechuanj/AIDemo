# -*- coding: utf-8 -*-

from fuzzy_cmeans import FuzzyCMeans
from distance import Method

# 6笔
samples = [
    [2, 12], [4, 9], [7, 13],
    [11, 5], [12, 7], [14, 4]
]

fcm = FuzzyCMeans()
fcm.distance_method = Method.Eculidean
fcm.convergence     = 0.001
fcm.max_iteration   = 20
fcm.m               = 3


fcm.add_center([5.0, 5.0], "Buy_Group")
fcm.add_center([10.0, 10.0], "Sell_Group")
fcm.add_center([12.0, 14.0], "Talk_Group")

for point in samples:
    fcm.add_sample(point)

fcm.setup()

def iteration_callback(iteration_times, groups):
    print ("iteration %r" % iteration_times)
    for group in groups:
        print("新旧群心 （%r) ： %r -> %r" % (group.tag, group.old_center, group.center))
        print ("-> %r" % group.samples)

def completion_callback(iteration_times, groups, sse):
    print ("completion %r and sse %r" % (iteration_times, sse))
    for group in groups:
        print("最终群心（%r）: %r, %r" % (group.tag, group.center, group.samples))

fcm.run(iteration_callback, completion_callback)

def classified_callback(point, group):
    print("%r classified to %r" % (point, group.tag))

fcm.classify([[2, 3], [3, 3], [5, 9],[10, 12]], classified_callback)