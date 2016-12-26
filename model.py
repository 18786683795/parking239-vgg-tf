import numpy as np
import tensorflow as tf

def conv2d(x, n_output, k_h=3, k_w=3, d_h=1, d_w=1, padding='SAME', activation=tf.nn.relu, name='conv2d', reuse=None):

	print("conv2d layer", x.get_shape())

	with tf.variable_scope(name or 'conv2d', reuse=reuse):
		W = tf.get_variable(
			name='W',
			shape=[k_h, k_w, x.get_shape()[-1], n_output],
			initializer=tf.contrib.layers.xavier_initializer_conv2d())

		#print W.get_shape()

		conv = tf.nn.conv2d(
			name='conv',
			input=x,
			filter=W,
			strides=[1, d_h, d_w, 1],
			padding=padding)

		b = tf.get_variable(
			name='b',
			shape=[n_output],
			initializer=tf.constant_initializer(0.0))

		h = tf.nn.bias_add(
			name='h',
			value=conv,
			bias=b)

		if activation:
			h = activation(h)

	return h, W

def linear(x, n_output, name=None, activation=tf.nn.relu, reuse=None):
	if len(x.get_shape()) != 2:
		x = flatten(x, reuse=reuse)

	n_input = x.get_shape().as_list()[1]

	print("Linear layer input=", n_input, "output=", n_output)

	with tf.variable_scope(name or "linear", reuse=reuse):
		W = tf.get_variable(
			name='W',
			shape=[n_input, n_output],
			dtype=tf.float32,
			initializer=tf.contrib.layers.xavier_initializer())

		b = tf.get_variable(
			name='b',
			shape=[n_output],
			dtype=tf.float32,
			initializer=tf.constant_initializer(0.0))

		preactivate = tf.nn.bias_add(
			name='h',
			value=tf.matmul(x, W),
			bias=b)

		tf.summary.histogram('pre_activations', preactivate)

		activations = preactivate

		if activation:
			activations = activation(preactivate)

		tf.summary.histogram('activations', activations)    

		return activations, W


def build_model(data_h, data_w, data_c, n_classes, learning_rate = 0.0001, model_name='parking-1', production_mode=False):
	activation_fn = tf.nn.relu #sigmoid

	X = tf.placeholder(tf.float32, shape=[None, data_h, data_w, data_c], name='X')
	Y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')
	
	if production_mode:
		keep_prob = 1.0
	else:	
		keep_prob = tf.placeholder(tf.float32)

	net = X
	
	net = conv2d(net, 64, 3, 3, name="conv2d_layer_1")[0]
	net = conv2d(net, 64, 3, 3, name="conv2d_layer_2")[0]
	net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	net = conv2d(net, 128, 3, 3, name="conv2d_layer_3")[0]
	net = conv2d(net, 128, 3, 3, name="conv2d_layer_4")[0]
	net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	net = conv2d(net, 256, 3, 3, name="conv2d_layer_5")[0]
	net = conv2d(net, 256, 3, 3, name="conv2d_layer_6")[0]
	net = conv2d(net, 256, 3, 3, name="conv2d_layer_7")[0]
	net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	net = conv2d(net, 512, 3, 3, name="conv2d_layer_8")[0]
	net = conv2d(net, 512, 3, 3, name="conv2d_layer_9")[0]
	net = conv2d(net, 512, 3, 3, name="conv2d_layer_10")[0]
	net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	net = conv2d(net, 512, 3, 3, name="conv2d_layer_11")[0]
	net = conv2d(net, 512, 3, 3, name="conv2d_layer_12")[0]
	net = conv2d(net, 512, 3, 3, name="conv2d_layer_13")[0]
	net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	net = conv2d(net, 512, 3, 3, name="conv2d_layer_14")[0]
	net = conv2d(net, 512, 3, 3, name="conv2d_layer_15")[0]
	net = conv2d(net, 512, 3, 3, name="conv2d_layer_16")[0]
	net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	# flatten
	shape = net.get_shape().as_list()
	print('shape before flatten', shape)
	net = tf.reshape(net, [-1, shape[1] * shape[2] * shape[3]])

	# fc
	net = linear(net, 4096, name='fc_layer_1', activation=activation_fn)[0]

	with tf.name_scope('dropout'):
		net = tf.nn.dropout(net, keep_prob)

	net = linear(net, 4096, name='fc_layer_2', activation=activation_fn)[0]

	with tf.name_scope('dropout'):
		net = tf.nn.dropout(net, keep_prob)

	# logits
	logits = linear(net, n_classes, name='fc_layer_logits', activation=activation_fn)[0]

	# softmax, resulting shape=[-1, n_classes]
	Y_pred = tf.nn.softmax(logits, name='softmax_layer')

	if production_mode:
		return X, Y_pred
	
	diff =  tf.nn.softmax_cross_entropy_with_logits(logits, Y)
	cross_entropy = tf.reduce_mean(diff, name='cross_entropy')

	correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	with tf.name_scope(model_name):
		tf.summary.scalar('cross_entropy', cross_entropy)
		tf.summary.scalar('train_accuracy', accuracy)
	summary = tf.summary.merge_all()

	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)		

	return X, Y_pred, Y, keep_prob, cross_entropy, optimizer, summary, accuracy

