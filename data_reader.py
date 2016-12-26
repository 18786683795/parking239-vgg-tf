import numpy as np
import csv
import ast
import os, sys
from random import shuffle
from data_classes import *
from PIL import Image

class Parking239:
	def __init__(self, data_path, limit=0):
		
		self.chunks = {}
		self.data_path = data_path

		for i, v in class_one_hot_by_name.items():
			print(v, i)
		
		self.chunks['train'] = []
		self.chunks['test'] = []

		self.build_set('train')
		self.build_set('test')

		print("training chunks                  ", len(self.chunks['train']) )
		print("testing chunks                   ", len(self.chunks['test']) )

		self.batch_index = {}
		self.batch_index['train'] = 0
		self.batch_index['test'] = 0

		self.normalize_data = False

		mean_path = os.path.join(self.data_path, 'mean.npy')                        # add m
		std_path = os.path.join(self.data_path, 'std.npy')

		if os.path.exists(mean_path) and os.path.exists(std_path):
			self.mean = np.load(mean_path)
			self.stddev = np.load(std_path)
			self.normalize_data = True
			print("Found mean, std files. Data shape=", self.mean.shape)


	def get_input_data_shape(self):
		if not self.normalize_data:
			raise "Data shape cannot be determined, calculate mean, std to continue...."
		return self.mean.shape	

	def build_set(self, set_name):
		self.chunks[set_name] = []
		with open(os.path.join(self.data_path, set_name + '.csv'), 'rt') as csvfile:
			reader = csv.reader(csvfile, delimiter=';', quotechar='|')
			for row in reader:
				image_relative_path, class_name = row
				class_one_hot = class_one_hot_by_name[class_name]
				
				image_path = os.path.join( self.data_path, image_relative_path )
				self.chunks[set_name].append([image_path, class_one_hot])

	def read_data(self, path):
		file_path = os.path.join(self.data_path, path)
		data = np.asfarray(Image.open(file_path))
		if self.normalize_data:
			data = (data-self.mean)/self.stddev
		return data

	def new_epoch(self, train_index = 0, test_index = 0):
		self.batch_index['train'] = train_index
		self.batch_index['test'] = test_index
		shuffle(self.chunks['train'])
		shuffle(self.chunks['test'])

		#np.random.shuffle(self.chunks['train'])
		#np.random.shuffle(self.chunks['test'])

	def get_next_batch_names(self, batch_size, batch_type='train'):
		total_size = len(self.chunks[batch_type])

		has_more_data = self.batch_index[batch_type] != total_size
		if not has_more_data:
			self.batch_index[batch_type] = 0
		
		num_chunks_left = total_size - self.batch_index[batch_type]
		size = min(num_chunks_left, batch_size)
		start = self.batch_index[batch_type]
		end = start  + size
		batch_filenames = self.chunks[batch_type][ start : end ]
		self.batch_index[batch_type] += size
		has_more_data = self.batch_index[batch_type] != total_size
		return batch_filenames, has_more_data

	def get_next_batch(self, batch_size, batch_type='train'):	
		batch_filenames, has_more_data = self.get_next_batch_names(batch_size, batch_type)
		batch = np.stack( [ [self.read_data(entry[0]), entry[1]] for entry in batch_filenames ] )
		#print(batch)
		xs = np.stack(batch[:,0])
		ys = np.stack(batch[:,1])
		return xs, ys, has_more_data
