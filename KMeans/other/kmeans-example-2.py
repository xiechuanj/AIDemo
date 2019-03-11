# -*- coding: utf-8 -*-
# sklearn.cluster.KMeans(n_clusters=8,
#                        init='k-means++',
#                        n_init=10,
#                        max_iter=300,
#                        tol=0.0001,
#                        precompute_distances='auto',
#                        verbose=0,
#                        random_state=None,
#                        copy_x=True,
#                        n_jobs=1,
#                        algorithm='auto'
#                        )
#
# 参数的意义：
#
# n_clusters:簇的个数，即你想聚成几类
# init: 初始簇中心的获取方法
# n_init: 获取初始簇中心的更迭次数，为了弥补初始质心的影响，算法默认会初始10次质心，实现算法，然后返回最好的结果。
# max_iter: 最大迭代次数（因为kmeans算法的实现需要迭代）
# tol: 容忍度，即kmeans运行准则收敛的条件
# precompute_distances：是否需要提前计算距离，这个参数会在空间和时间之间做权衡，如果是True 会把整个距离矩阵都放到内存中，auto 会默认在数据样本大于featurs*samples 的数量大于12e6 的时候False,False 时核心实现的方法是利用Cpython 来实现的
# verbose: 冗长模式（不太懂是啥意思，反正一般不去改默认值）
# random_state: 随机生成簇中心的状态条件。
# copy_x: 对是否修改数据的一个标记，如果True，即复制了就不会修改数据。bool 在scikit-learn 很多接口中都会有这个参数的，就是是否对输入数据继续copy 操作，以便不修改用户的输入数据。这个要理解Python 的内存机制才会比较清楚。
# n_jobs: 并行设置
# algorithm: kmeans的实现算法，有：‘auto’, ‘full’, ‘elkan’, 其中 'full’表示用EM方式实现


import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def jieba_tokenize(text):
    return jieba.lcut(text)

tfidf_vectorizer = TfidfVectorizer(tokenizer=jieba_tokenize,lowercase=False)
'''
tokenizer: 指定分词函数
lowercase: 在分词之前将所有的文本转换成小写，因为涉及到中文文本处理，所以最好是False
'''

text_list = ["天气真好","今天天气真好啊啊啊啊","小明上了清华大学","我今天拿到了Google的Offer","清华大学在自然语言处理方面真历害", "Offer"]
# 需要进行聚类的文本集
tfidf_matrix = tfidf_vectorizer.fit_transform(text_list)

num_clusters = 3
km_cluster = KMeans(n_clusters=num_clusters, max_iter=300, n_init=40, init='k-means++', n_jobs=-1)
'''
n_clusters: 指定K的值
max_iter: 对于单次初始值计算的最大迭代次数
n_init: 重新选择初始值的次数
init: 制定初始值选择的算法
n_jobs: 进程个数，为-1的时候是指默认跑满CPU
注意，这个对于单个初始值的计算始终只会使用单进程计算，
并行计算只是针对与不同初始值的计算。比如n_init=10，n_jobs=40, 
服务器上面有20个CPU可以开40个进程，最终只会开10个进程
'''
# 返回各自文本的所被分配到的类索引
result = km_cluster.fit_predict(tfidf_matrix)
print("Predicting result: ", result)