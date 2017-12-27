import cv2
import numpy as np
import glob
import sys
sys.path.append('C:\Program Files\Python36\Lib\site-packages\libsvm-3.22\python')
from svmutil import *
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import datasets	
from sklearn import svm

C = 12.5
gamma = 0.50625

train_dir = 'D:\workspace\TrafficSignRecognitionAndDetection\Dataset\Train\GTSRB_Final_Training_Images\Final_Training\Images'
# train_dir = 'D:\workspace\TrafficSignRecognitionAndDetection\Train\Datasets'
test_dir = 	'D:\workspace\TrafficSignRecognitionAndDetection\Dataset\Test\GTSRB_Final_Test_Images\Final_Test\Images'
svm_model_file = 'svm_model.xml'
train_data_file = 'train_data'
test_data_file = 'test_data'

width = height = 48
hog = cv2.HOGDescriptor(_winSize = (width, height),
						_blockSize = (width // 2, height // 2),
						_blockStride = (width // 4, height // 4),
						_cellSize = (width // 2, height // 2),
						_nbins = 9,
						_derivAperture = 1,
						_winSigma = -1,
						_histogramNormType = 0,
						_L2HysThreshold = 0.2,
						_gammaCorrection = 1,
						_nlevels = 64, 
						_signedGradient = True)

def load_datasets(_dir, _images, _labels):
	# load all images and labels in the dir directory based on the .csv file
	csv_dir = glob.glob(_dir + '\*.csv')[0]	

	with open(csv_dir, 'r') as f:
		# get the number of .ppm file	
		n_datasets = f.read().count('ppm')
		print('\tn datasets:', n_datasets)
		print('\tProcessing...')
		f.seek(0, 0)
		
		# skip the first line of .csv
		f.readline()
		# read each line in the file
		for line in f:
			filename, w, h, x1, y1, x2, y2, classId = line.replace('\n', '').split(';')
			img = cv2.imread(_dir + '\\' + filename, 0)
			# crop and scale the image
			img = img[int(y1):int(y2), int(x1):int(x2)]	
			img = cv2.resize(img, (width, height))
			_images.append(img)
			_labels.append(int(classId))

def calculate_hog(_images, _data_file):
	print('\tProcessing...')
	n_datasets = len(_images)
	
	# open _data_file and append hog descriptors to it
	with open(_data_file, 'a') as f:
		# calculate hog_descriptors for each image in _images
		for img in _images:
			descriptor = hog.compute(img)
			f.write(' '.join(str(x[0]) for x in descriptor) + '\n')
	
def load_data_file(_data_file):
	hog_descriptors = []
	labels = []
	
	with open(_data_file, 'r') as f:
		# load labels
		for x in f.readline().split(' '):
			labels.append(int(x))
		# load hog descriptors
		for line in f:
			descriptor = []
			for x in line.split(' '):
				descriptor.append([x])
			hog_descriptors.append(np.array(descriptor, dtype=np.float32))

	return hog_descriptors, labels

'''	
Execute another function and calculate the execution time of that
Parameters:
	- notification: the string that printed when the function executed 
	- func: the function name
	- *arg: all parameters of the function
'''
def execute(_notification, _func, *_args):	
	print(_notification)
	
	t_start = datetime.now()
	result = _func(*_args)
	delta = datetime.now() - t_start

	print('Time: %fs' % (delta.seconds + delta.microseconds/1E6))
	print('-----')
	return result