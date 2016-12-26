import os, sys
import numpy as np
import tensorflow as tf
from random import shuffle
from model import build_model
from data_reader import Parking239
from data_classes import *

from config import CONFIG

DATASET_WORKING_DIR = CONFIG['DATASET_WORKING_DIR']

n_classes = len(classes)

learning_rate = 0.0001
max_epochs = 5000
max_steps = 0 #10000
batch_size = 100

DATA_LIMIT = 0

validation_step = 10                   # validate the model each N steps
train_keep_probability = 0.5           # dropout

model_name = 'parking-1'               # this is used for tensoboard

train_model = True
save_model  = True                             # save the model
save_model_step = 100                          # save the mode each N steps
model_path = './model/' + model_name           # where to save it
init_step = 0 
#init_load_model_from = './model/doesitworknow-1000' # model_path + "-1000" #./model/m5-' + str(init_step)
init_load_model_from = ''


if __name__ == "__main__":

	print(" (loading metadata...) ")
	dataset = Parking239(DATASET_WORKING_DIR, limit=DATA_LIMIT)

	# read the mean, stddev and determine the data shape
	mean_shape = dataset.get_input_data_shape()
	data_h, data_w, data_c = mean_shape
	print("Mean shape", mean_shape)


	g = tf.Graph()
	with g.as_default():
	
		with tf.Session(graph=g) as sess:

			X, Y_pred, Y, keep_prob, cost, optimizer, summary, accuracy = build_model(data_h, data_w, data_c, n_classes, learning_rate, model_name=model_name)

			init = tf.global_variables_initializer()
			sess.run(init)

			train_writer = tf.summary.FileWriter('./summary/train', sess.graph)
			validation_writer = tf.summary.FileWriter('./summary/test')

			saver = tf.train.Saver()
			if len(init_load_model_from) > 0:
				print("LOADING MODEL FROM", init_load_model_from)
				saver.restore(sess, init_load_model_from)

			else:	
				print("INITIALIZING MODEL (starting training from scratch)")
			
			step = init_step + 1
			for epoch in range(max_epochs):
				print("epoch", epoch, "of", max_epochs)
				dataset.new_epoch()
				has_more_data = True
				while has_more_data:
					
					if train_model:
						batch_xs, batch_ys, has_more_data = dataset.get_next_batch(batch_size)
						calculated_cost, train_summary, _ = sess.run([cost, summary, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: train_keep_probability})
						train_writer.add_summary(train_summary, step)
					
						print("Training epoch", epoch+1, "step", step, "batch ", batch_xs.shape, batch_ys.shape, "calculated_cost", calculated_cost)

					if not train_model or step % validation_step == 0:
						batch_xs, batch_ys, _  = dataset.get_next_batch(batch_size, batch_type='test')
						calculated_cost, validation_summary, calculated_accuracy = sess.run([cost, summary, accuracy], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0})
						if train_model:
							validation_writer.add_summary(validation_summary, step)	

						print("Validation step", step, "batch",  "cost", calculated_cost, "accuracy", calculated_accuracy)	

					if train_model and save_model and step % save_model_step == 0:
						save_path = saver.save(sess, model_path, global_step=step)
						print("Model saved in file: %s" % save_path)

					step = step + 1
					if max_steps > 0 and step >= max_steps:
						break

				if max_steps > 0 and step >= max_steps:
					break			
