import tensorflow as tf


with tf.variable_scope("foo"):
    v = tf.get_variable(
        "v",[1],initializer=tf.constant(1.0)
    )

#this illegal
with tf.variable_scope("foo"):
    v = tf.get_variable("v",[1])

#直接获取已经声明对变量
with tf.variable_scope("foo",reuse=True):
    v1 = tf.get_variable("v",[1])
    print(v == v1)

#reuse == true 只能获取已经创建对变量,报错，该命名空间不存在
with tf.variable_scope("bar",reuse=True):
    v = tf.get\("v",[1])

