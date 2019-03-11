# -*- coding: utf-8 -*-

import numpy as np
from enum import Enum

class Method(Enum):
    Eculidean          = 0   # 欧式距离
    Cosine_Similarity  = 1   # 余弦相似

class Distance:

    def calculate(self, x1=[], x2=[], method=Method.Eculidean):
        return {
            Method.Eculidean: self.eculidean,
            Method.Cosine_Similarity: self.cosine_similarity
        }.get(method)(x1, x2)

    # x1, x2 are vectors
    # 欧式距离越小越近
    def eculidean(self, x1, x2):
        sum = 0.0
        for i, a in enumerate(x1):
            b = x2[i]
            sum += (a - b) ** 2
        return np.sqrt(sum)

    # Cosine 相似度是越大越近， 故最近计算完的相似度要再用 1.0 去减，就会越小越近
    def cosine_similarity(self, x1, x2):
        sum_a = 0.0
        sum_b = 0.0
        sum_ab = 0.0
        for i, a in enumerate(x1):
            b      = x2[i]
            sum_a  += a ** 2
            sum_b  += b ** 2
            sum_ab += a * b

        similarity = sum_ab / ((sum_a * sum_b) ** 2) if(sum_a * sum_b > 0.0) else 0.0
        return 1.0 - similarity