import tensorflow as tf

# 声明w1, w2两个变量。这里还通过seed参数设定了随机种子，
# 这样可以保证每次运行得到的结果是一样的
w1 = tf.Variable(tf.random_normal((2,3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3,1), stddev=1, seed=1))

# 暂时将输入的特征向量定义为一个常量。注意这里x是一个1x2的矩阵
x = tf.constant([[0.7, 0.9]])

# 前向传播算法获得神经网络的输出
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    # 初始化w1, w2
    # sess.run(w1.initializer)
    # sess.run(w2.initializer)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 输出
    print(sess.run(y))