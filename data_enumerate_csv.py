import os, sys, csv, math
import numpy as np

from PIL import Image
from random import shuffle
from config import CONFIG

DATASET_WORKING_DIR = CONFIG['DATASET_WORKING_DIR']
n_stat = CONFIG['DATASET_MEAN_STD_NUMBER']
image_h, image_w = CONFIG['TARGET_IMAGE_SIZE']


data = []

print("Scanning images....")
for class_dir in os.listdir(DATASET_WORKING_DIR):
	class_path = os.path.join(DATASET_WORKING_DIR, class_dir)
	if not os.path.isdir(class_path):
		continue

	for file in os.listdir(class_path):
		data.append([os.path.join(class_dir, file), class_dir])

shuffle(data)

end_training = int(len(data)*0.8)

index = 0

print("Creating training set....")
with open(os.path.join(DATASET_WORKING_DIR, 'train.csv'), 'wt') as csvfile:
	writer = csv.writer(csvfile, delimiter=';', quotechar='|')
	
	while index < end_training:
		writer.writerow(data[index])
		index += 1

print("Creating test set....")
with open(os.path.join(DATASET_WORKING_DIR, 'test.csv'), 'wt') as csvfile:
	writer = csv.writer(csvfile, delimiter=';', quotechar='|')
	
	while index < len(data):
		writer.writerow(data[index])
		index += 1


n_load = min(end_training, n_stat)
images = np.empty([n_load, image_h, image_h, 3])
print("Calculating mean, stddev of the first ", n_load, "images")
for i in range(n_load):
	sys.stdout.write("loading image " +str(i+1) + " of " + str(n_load) + "\n")
	#print("loading image ", (i+1), "of", n_load, end="\r")
	file_path = os.path.join(DATASET_WORKING_DIR, data[i][0])
	im = np.asfarray(Image.open(file_path))
	images[i] = im

print("\ncalculating mean....")
np.save(os.path.join(DATASET_WORKING_DIR, 'mean.npy'), np.mean(images, axis=0))

print("calculating std dev....")
np.save(os.path.join(DATASET_WORKING_DIR, 'std.npy'), np.std(images, axis=0))


