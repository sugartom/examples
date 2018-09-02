import tensorflow as tf

class Mnist:
	def __init__(self):
		self.image = tf.placeholder('float',shape=[1, 28, 28, 1],name='input_image')
		x = self.image
		with tf.variable_scope('layer0'):
			W = tf.get_variable("weights", shape = [5, 5, 1, 10])
			b = tf.get_variable("bias", shape = [10])
			x = tf.nn.conv2d(x, W, [1, 1, 1, 1], "VALID")
			x = tf.nn.bias_add(x, b)
			x = tf.nn.max_pool(x, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
			x = tf.nn.relu(x)
		with tf.variable_scope('layer1'):
			W = tf.get_variable("weights", shape = [5, 5, 10, 20])
			b = tf.get_variable("bias", shape = [10])
			x = tf.nn.conv2d(x, W, [1, 1, 1, 1], "VALID")
			x = tf.nn.bias_add(x, b)   
			x = tf.nn.max_pool(x, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
			x = tf.nn.relu(x)
		with tf.variable_scope('layer2'):
			x = tf.reshape(x, [-1, 320])
		with tf.variable_scope('layer3'):
			W = tf.get_variable("weights", shape = [320, 50])
			b = tf.get_variable("bias", shape = [50])
			x = tf.matmul(x, W)
			x = tf.add(x, b)
			x = tf.nn.relu(x)
		with tf.variable_scope('layer4'):
			W = tf.get_variable("weights", shape = [50, 10])
			b = tf.get_variable("bias", shape = [10])
			x = tf.matmul(x, W)
			x = tf.add(x, b)
			x = tf.nn.log_softmax(x)
		self.classifier = x