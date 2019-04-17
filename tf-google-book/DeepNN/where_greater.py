import tensorflow as tf
from numpy.random import RandomState

v1 = tf.constant([1, 2, 3, 4])
v2 = tf.constant([4, 3, 2, 1])

batch_size = 8

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# single layer

w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

loss_less = 10
loss_more = 1

loss = tf.reduce_sum(tf.where(tf.greater(y, y_),
                              (y - y_) * loss_more,
                              (y_ - y) * loss_less))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 两个正确值，再加上一个不可预测的噪音，一般噪声为一个平均值为0的小量

Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]

# To train

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step,
                 feed_dict={x: X[start:end], y_: Y[start:end]})
    print(sess.run(w1))

global_step = tf.Variable(0)

# decay to generate learning rate

learing_rate = tf.train.exponential_decay(
    0.1, global_step, 100, 0.96, staircase=True)
learing_step = tf.train.GradientDescentOptimizer(learing_rate).minimize(loss, global_step=global_step)

w = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w)
lam = 0.05
loss = tf.reduce_mean(tf.square(y_ - y)) + tf.contrib.layers.l2_regularizer(lam)(w)

#example
weights = tf.constant([1,2],[-3,4])
with tf.Session() as sess:
    print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))
    print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))