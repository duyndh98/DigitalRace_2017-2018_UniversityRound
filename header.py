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
from matplotlib.colors import ListedColormap

C = 12.5
gamma = 0.50625

train_dir = 'D:\workspace\TrafficSignRecognitionAndDetection\Dataset\Train\GTSRB_Final_Training_Images\Final_Training\Images'
test_dir = 	'D:\workspace\TrafficSignRecognitionAndDetection\Dataset\Test\GTSRB_Final_Test_Images\Final_Test\Images'
templates_dir = 'D:\workspace\TrafficSignRecognitionAndDetection\Train\\templates'

svm_model_file = '_' + str(C) + '_' + str(gamma) + '_svm_model.xml'
confusion_matrix = '_' + str(C) + '_' + str(gamma) + '_confusion_matrix.png'
visualize = '_' + str(C) + '_' + str(gamma) + '_visualize.png'

train_data_file = 'train_data'
test_data_file = 'test_data'
video_input = 'video_input.avi'
video_output = 'video_output.avi'
output = 'output.txt'

# threshold
lower_red1 = np.array([0, 60, 60])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 60, 60])
upper_red2 = np.array([179, 255, 255])

lower_blue = np.array([105, 60, 60])
upper_blue = np.array([130, 255, 255])


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

def load_templates():
	images = []
	for img_dir in glob.glob(templates_dir + '\*'):
		img = cv2.imread(img_dir)
		images.append(img)
	return images

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
	print('\tLoading data file')
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

def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')