from header import *

def svm_visualize(_svm, _hog_descriptors, _labels):
	print('\tProcessing...')
	X = np.array(_hog_descriptors)[:, 0:]  # we only take the first two features.
	y = np.array([[x] for x in _labels])
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s = 20, edgecolors = 'k')
   
	h = 0.01  # step size in the mesh
	# create a mesh to plot in
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	
	plt.title('SVC with RBF kernel')
	
	# init SVC with RBF kernel
	clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)
	# SVM classify with 2-dimensional hog decriptors vector
	clf.fit(X[:,:2,0], y.ravel())
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.7)

	plt.savefig(visualize)
	plt.show()
	
def load_train_datasets():
	train_images = []
	train_labels = []
	
	# load all images and labels in the train directory
	for class_dir in glob.glob(train_dir + '\*'):
		print('\t' + class_dir)
		load_datasets(class_dir, train_images, train_labels)

	# open train data file and write labels to it
	with open(train_data_file, 'w') as f:
		f.write(' '.join(str(x) for x in train_labels) + '\n')

	return train_images

def svm_training():
	# load hog descriptors and labels from train data file
	train_hog_descriptors, train_labels = load_data_file(train_data_file)
	print('\tProcessing...')

	# init SVM model
	svm = cv2.ml.SVM_create()
	svm.setC(C)
	svm.setGamma(gamma)	
	svm.setType(cv2.ml.SVM_C_SVC)
	svm.setKernel(cv2.ml.SVM_RBF)

	# Train SVM on train data  
	svm.train(np.array(train_hog_descriptors), cv2.ml.ROW_SAMPLE, np.array(train_labels))
	# Save trained model
	svm.save(svm_model_file)

	return svm, train_hog_descriptors, train_labels

def train():
	'''
	# After the first running, you can skip these steps at the next time
	# by loading train_data file in svm_training() function
	train_images = execute('Loading train datasets', load_train_datasets)
	execute('HOG calculating', calculate_hog, train_images, train_data_file)
	'''
	svm, train_hog_descriptors, train_labels = execute('Training', svm_training)
	# execute('SVM visualize', svm_visualize, svm, train_hog_descriptors, train_labels)