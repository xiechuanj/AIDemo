# -*- coding: utf-8 -*-
import tensorflow as tf

from numpy.random import RandomState

# 定义训练数据batch的大小
batch_size = 8



# 两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
# 回归问题一般只胡一个输出节点
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

# 定义了一个单层的神经网络前向传播的过程，这里就是简单的加权和。
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义预测多了和预测少了的成本。
loss_less = 1
loss_more = 10
loss = tf.reduce_sum(tf.where(tf.greater(y, y_),
                              (y - y_) * loss_more,
                              (y_ - y) * loss_less))
# 反向传播算法
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

# 定义规则来给出样本的标签。在这里所有 xl+x2<1 的样例都被认为是正样本（ 比如i零件合格）， ＃而其他为负样本（ 比如l骂：件不合格）。和 TensorFlow 游乐场中的表示法不大一样的地方是， ＃在这里使用 0 米表示负样本， 1 来表示正样本。大部分解决分类问题的神经网络都会采用 # 0 和 1 的农尽方法。
Y = [[x1 + x2 + rdm.rand()/10.0 - 0.05]  for (x1, x2) in X]


with tf.Session() as sess:
    # 初始化w1, w2
    # sess.run(w1.initializer)
    # sess.run(w2.initializer)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)


    # 设定训练的轮数
    STEPS = 5001
    for i in range(STEPS):
        # 每次选取 batch_size 个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)

        # 通过选取的样本训练神经网络并更新参数。
        sess.run(train_step,
                 feed_dict={x: X[start:end], y_: Y[start:end]})


    print(sess.run(w1))


