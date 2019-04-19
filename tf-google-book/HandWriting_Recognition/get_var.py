import tensorflow as tf

#定义前向传播过程--优化后

def inference(input_tensor,reuse=False):
    #first layer
    with tf.variable_scope('layer1',reuse=reuse):
        "根据reuse判断是否创建好"
        weights = tf.get_variable("weights",(INPUT_NODE,LAYER!_NODE),
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases",[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)

    with tf.variable_scope('layer2',reuse=reuse):
        "根据reuse判断是否创建好"
        weights = tf.get_variable("weights",(INPUT_NODE,LAYER!_NODE),
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases",[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1,weights)+biases
    return layer2


x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
y = inference(x)




