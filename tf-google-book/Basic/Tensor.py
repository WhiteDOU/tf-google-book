import tensorflow as tf

tf.enable_eager_execution()
a = tf.constant([1,2],name="a")
b = tf.constant([3,4],name="b")
result = tf.add(a, b, name="add")
print(result)
#sess = tf.Session()
#sess.run(result)

#sess.close()

#with tf.Session() as sess:
    #sess.run(result)


