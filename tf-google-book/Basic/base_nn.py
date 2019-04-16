import tensorflow as tf
from numpy.random import RandomState
import numpy as np

# batch size
batch_size = 8

# define nn
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 在 shape 的一个维度上使用 None 可以方便使用不同的 batch 大小。在训练时需要把数据分
# 成比较小的 batch， 但是在测试时，可以一次性使用全部的数据。当数据集比较小时这样比较
# 方便测试，但数据集比较大时，将大量数据放入一个 batch 吁能会导致内存溢出。

x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

# front
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# loss function
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
    + (1 - y) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# to create a dataset

rdm = RandomState(1)
dataset_size = 128

X = rdm.rand(dataset_size, 2)
Y = [(int(x1 + x2 < 1)) for (x1, x2) in X]

# Session
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    # init
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))

    # epochs
    EPOCHES = 5000
    for i in range(EPOCHES):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        # tot train
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X,y_:Y})
            #print("After %d training step (s) , cross entropy on all data is %g " % (i, total_cross_entropy))
    print(sess.run(w1))
    print(sess.run(w2))

