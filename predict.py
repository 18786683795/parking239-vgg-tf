import os, sys, math
import numpy as np
import tensorflow as tf

from config import CONFIG
from subprocess import call
from random import shuffle
from model import build_model
from data_classes import *
from PIL import Image

n_classes = len(classes)

model_name                 = CONFIG['PREDICT_MODEL_NAME']
directory_of_mean_and_dev  = CONFIG['MEAN_STDEV_DIRECTORY']
target_image_size 		   = CONFIG['TARGET_IMAGE_SIZE']

init_load_model_from       = os.path.join('model', model_name)

if __name__ == "__main__":

	jpgfile = sys.argv[1]
	if jpgfile.startswith("file://"):
		jpgfile = jpgfile[7:]
	
	if not os.path.exists(jpgfile):
		raise "input wave file not found"

	mean = None
	stddev = None
	normalize_data = False

	mean_path = os.path.join(directory_of_mean_and_dev, 'mean.npy')
	std_path = os.path.join(directory_of_mean_and_dev, 'std.npy')
	if os.path.exists(mean_path) and os.path.exists(std_path):
		mean = np.load(mean_path)
		stddev = np.load(std_path)
		normalize_data = True
		print("Found mean, std files. Data shape=", mean.shape)
	else:
		raise "Could not read man/stddev files"

	# read the mean, stddev and determine the data shape
	data_shape = mean.shape
	data_h, data_w, data_c = data_shape
	print("Mean shape", data_shape)

	image = Image.open(jpgfile)
	image = image.resize(target_image_size, Image.ANTIALIAS)

	data = np.asfarray(image)
	if normalize_data:
		data = (data-mean)/stddev

	g = tf.Graph()
	with g.as_default():
		with tf.Session(graph=g) as sess:
			X, Y_pred = build_model(data_h, data_w, data_c, n_classes, production_mode=True)

			init = tf.global_variables_initializer()
			sess.run(init)

			saver = tf.train.Saver()
			if len(init_load_model_from) > 0:
				print("LOADING MODEL FROM", init_load_model_from)
				saver.restore(sess, init_load_model_from)
			else:
				print("Can't read the model")
				raise "Can't read the model"
				

			batch_xs = np.stack([data])

			predicted_one_hot = sess.run([Y_pred], feed_dict={X: batch_xs})[0]
			result_dev = np.std(predicted_one_hot)
			if result_dev < 0.001:
				print("offset", start, "I don't know what is that!")							
			else:
				class_index = np.argmax(predicted_one_hot)
				confidence = predicted_one_hot[0, class_index]
				print("prediction", classes[class_index].rjust(15), "{:.2%}".format(confidence), "%", "dev", result_dev)	
