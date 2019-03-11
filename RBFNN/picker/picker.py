# -*- coding: utf-8 -*-
from kmeans.center import Choice
from kmeans.kmeans import KMeans
from sample import Sample

class Picker(object):

    # 结合 K-Means 里的方法来挑选隐藏层中心点
    kmeans = KMeans()
    samples = []  # <Sample Object>

    # Random Picking
    def shuffing(self, k):
        maker    = self.kmeans.center_maker
        maker.samples = self.samples
        return maker.shuffing(k)

    # K-Means++ 方法挑选后做聚类更新中心点（预设欧式距离）
    def clustering(self, k):
        # 先把 Sample Object 的 features 都取出来做聚类（Clustering）
        for sample in self.samples:
            self.kmeans.add_sample(sample.features)
        kmeans = self.kmeans
        kmeans.center_choice = Choice.Plus
        kmeans.make_centers(k)
        kmeans.max_iteration = 100
        kmeans.convergence = 0.001
        kmeans.setup()
        kmeans.run()
        # 将聚好且修正学习过好的中心点重新做回去 <Sample Object>，回传回去给 RBFNN 使用
        clustered_samples = []  # <Sample Object>
        for group in kmeans.groups:
            sample = Sample(group.center) # 这里的 Sample Object 不需要设定 targets
            clustered_samples.append(sample)
        return clustered_samples

    # @property
    # def samples(self):
    #     return self.maker.samples
    #
    # @samples.setter
    # def samples(self, value):
    #     self.maker.samples = value


