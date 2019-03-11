# -*- coding: utf-8 -*-

from rbfnn import RBFNN, PickMethod
from sample import Sample

# 17笔
training_features = [
    # Target: [1, 0]
    [5, 4], [3, 4], [2, 5], [1, 1], [1, 2], [2, 2], [3, 2], [3, 1],
    # Target: [0, 1]
    [6, 4], [7, 6], [5, 6], [6, 5], [7, 8],
    # Target: [0, 0]
    [3, 12], [5, 20], [3, 20]
]

training_targets = [
    [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
    [0, 1], [0, 1], [0, 1], [0, 1], [0, 0],
    [0, 0], [0, 0], [0, 0], [0, 0]
]

# 制作训练样本
samples = [] # <Sample Object>
for index, features in enumerate(training_features):
    sample = Sample(features, training_targets[index])
    samples.append(sample)

rbfnn = RBFNN()
rbfnn.add_samples(samples)
rbfnn.initialize_centers(3, PickMethod.Clustering) # PickMethod.Random
rbfnn.initialize_outputs()
rbfnn.randomize_weights(-0.1, 0.1)
rbfnn.max_iteration = 5000

def iteration_callback(network):
    print ("iteration %r, cost %r" % (network.iteration, network.cost.rmse))

def completion_callback(network, success):
    if success == True:
        print("Training Succeed %r, cost %r" % (network.iteration, network.cost.rmse))
        print("Predicating ...")
        print("Predicated Outputs %r vs [1, 0]" % network.predicate([3, 4]))
        print("Predicated Outputs %r vs [0, 1]" % network.predicate([6, 5]))
        print("Predicated Outputs %r vs [0, 0]" % network.predicate([7, 8]))
    else:
        print("Training Failed %r, cost %r" % (network.iteration, network.cost.rmse))

rbfnn.training(iteration_callback, completion_callback)
