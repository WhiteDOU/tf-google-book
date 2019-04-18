from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

(train_imgae, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
print(train_labels.size)
print(test_labels.size)

mnist = input_data.read_data_sets("/Users/white/Documents/OneDrive\ -\ leverage\ proactive\ deliverables/GitHub/tf-google-book",one_hot=True)

print(mnist.train.num_examples)
print(mnist.train.labels[0])

batch_size = 100
#choose next batch
xs,ys = mnist.train.next_batch(batch_size)
print(xs.shape)

