import tensorflow as tf

from numpy.random import RandomState

# 定义训练数据batch的大小
batch_size = 8

# 声明w1, w2两个变量。这里还通过seed参数设定了随机种子，
# 这样可以保证每次运行得到的结果是一样的
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1))

# 定义placeholder 作为存放输入数据的地方。这里维度也不一定要定义
# 但如果维度是确定的，那么给出维度可以降低出错的概率
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")
# 前向传播算法获得神经网络的输出
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播的算法
y = tf.sigmoid(y)

# 定义交叉熵
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(1-y, 1e-10, 1.0))
    + (1-y) * tf.log(tf.clip_by_value(1-y, 1e-10, 1.0))
)

# 反向传播算法
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

# 定义规则来给出样本的标签。在这里所有 xl+x2<1 的样例都被认为是正样本（ 比如i零件合格）， ＃而其他为负样本（ 比如l骂：件不合格）。和 TensorFlow 游乐场中的表示法不大一样的地方是， ＃在这里使用 0 米表示负样本， 1 来表示正样本。大部分解决分类问题的神经网络都会采用 # 0 和 1 的农尽方法。
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]


with tf.Session() as sess:
    # 初始化w1, w2
    # sess.run(w1.initializer)
    # sess.run(w2.initializer)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run(w1))
    print(sess.run(w2))
    # # 输出
    # print(sess.run(y, feed_dict={x: [[0.7, 0.9],[0.1, 0.4], [0.5, 0.8]]}))

    # 设定训练的轮数
    STEPS = 5001
    for i in range(STEPS):
        # 每次选取 batch_size 个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)

        # 通过选取的样本训练神经网络并更新参数。
        sess.run(train_step,
                 feed_dict={x: X[start:end], y_: Y[start:end]})

        if  i % 1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy =sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("Atfer %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))

