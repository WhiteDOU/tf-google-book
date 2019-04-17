import tensorflow as tf
v1 = tf.Variable(0,dtype=tf.float32)

#steps
step = tf.Variable(0,trainable=False)

#定义一个类，并且初始化给定衰减率 0。99

ema = tf.train.ExponentialMovingAverage(0.99,step)

#定义一个滑动操作，需要给定一个列表
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run([v1,ema.average(v1)]))

    sess.run(tf.assign(v1,5))

    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))
    #更新step
    sess.run(tf.assign(step,10000))
    sess.run(tf.assign(v1,10))
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))
    #z再次更新
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))