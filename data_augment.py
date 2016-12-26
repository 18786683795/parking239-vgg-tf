# augments images by rotating, cropping them
# finally resizes the images to squeares

import os, sys
import numpy as np
import random
from PIL import Image
from config import CONFIG

DATA_SOURCE_DIR = CONFIG['DATASET_SOURCE_DIR']
DATASET_WORKING_DIR = CONFIG['DATASET_WORKING_DIR']
target_image_size = CONFIG['TARGET_IMAGE_SIZE']

original_size = CONFIG['ORIGINAL_IMAGE_SIZE']

#crops = [(200, 130, original_size[0]-200, original_size[1] - 130),  (546, 327, 415, 224), (524, 467, 299, 445)]
crop_amount = 0.8
num_crops = 4

rotations = [-15, -10, -5, 5, 10, 15]

if not os.path.exists(DATASET_WORKING_DIR):
	os.makedirs(DATASET_WORKING_DIR)

apply_augmentation_to_subfolders = ['free', 'occupied', 'closed']

images = []
count = 0


def save_image(image, filename):
	resized = image.resize(target_image_size, Image.ANTIALIAS)
	outfile = os.path.join(DATASET_WORKING_DIR, subfolder, filename)
	resized.save(outfile, "JPEG")


for subfolder in os.listdir( DATA_SOURCE_DIR ):
	if subfolder == 'unsorted':
		continue
		
	target_subfolder = os.path.join(DATASET_WORKING_DIR, subfolder)
	if not os.path.exists( target_subfolder ):
		os.makedirs( target_subfolder )
	for file in os.listdir( os.path.join(DATA_SOURCE_DIR, subfolder)):
		filename = os.path.splitext(file)[0]
		
		#print("file", file, os.path.splitext(file))
		print("processing ", count, "\r", end="")
		print("")
		count += 1
		ext = os.path.splitext(file)[1]
		
		if ext == '.jpg' or ext == '.jpeg' or ext == '.JPG':
			file_path = os.path.join(DATA_SOURCE_DIR, subfolder, file)
			im = Image.open(file_path)
			save_image(im, filename + "_original.jpg")

			if True: #subfolder in apply_augmentation_to_subfolders:

				for rotation_index, rotation in enumerate(rotations):
					rotated = im.rotate(rotation, resample=Image.BICUBIC)
					save_image(rotated, filename + "_original"  + "_rot" + str(rotation_index) + ".jpg")

				for crop_index in range(num_crops):
					x = random.randint(0, int(original_size[0]*(1 - crop_amount)))
					y = random.randint(0, int(original_size[1]*(1 - crop_amount)))
					w = original_size[0] - x
					h = original_size[1] - y

					crop_size = (x,y,w,h)

					cropped = im.crop(crop_size)
					save_image(cropped, filename + "_crop" + str(crop_index) + ".jpg")
					
					for rotation_index, rotation in enumerate(rotations):
						rotated = cropped.rotate(rotation, resample=Image.BICUBIC)
						save_image(rotated, filename + "_crop" + str(crop_index) + "_rot" + str(rotation_index) + ".jpg")
