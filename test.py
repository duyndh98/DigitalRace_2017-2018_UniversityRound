from header import *

def calculate_confusion_matrix(_predictions, _labels):
    print('\tConfusion matrix')
    confusion = np.zeros((43, 43), np.int32)
    for i, j in zip(_labels, _predictions):
        confusion[int(i), int(j)] += 1

    norm_confusion = []
    
    for row in confusion:
        tmp = []
        for x in row:
            tmp.append(float(x) / float(sum(row)))
        norm_confusion.append(tmp)

    width, height = len(confusion[0]), len(confusion)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_confusion), cmap=plt.cm.jet, interpolation='nearest')

    for x in range(width):
        for y in range(height):
            ax.annotate(str(confusion[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = [str(x) for x in range(len(confusion))]
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.xlabel('Predictions')
    plt.ylabel('Labels')

    plt.title('Confusion matrix')
    # plt.show()
    plt.savefig(confusion_matrix)

def svm_evaluation(_predictions, _labels):
	accuracy = (np.array(_labels) == _predictions.astype(int)).mean()
	print('\tPercentage Accuracy: %.2f %%' % (accuracy * 100))

	calculate_confusion_matrix(_predictions, _labels)

def load_test_datasets():
	test_images = []
	test_labels = []

	# load all images and labels in the test directory
	load_datasets(test_dir, test_images, test_labels)

	# open test_data_file and write labels to it
	with open(test_data_file, 'w') as f:
		f.write(' '.join(str(x) for x in test_labels) + '\n')

	return test_images

def svm_testing(_svm):
	print('\tProcessing...')
	
	# load hog descriptors and labels from test data file
	test_hog_descriptors, test_labels = load_data_file(test_data_file)
	
	predictions = _svm.predict(np.array(test_hog_descriptors))[1].ravel().astype(int)
	return predictions, test_labels

def test():
	'''
	# After the first running, you can skip these steps at the next time 
	# by loading test_data file in svm_testing() function
	test_images = execute('Loading test datasets', load_test_datasets)
	execute('HOG calculating', calculate_hog, test_images, test_data_file)
	'''
	# load svm model from svm model file
	svm = execute('Loading model', cv2.ml.SVM_load, svm_model_file)
	# test
	predictions, test_labels = execute('Testing', svm_testing, svm)
	execute('Evaluating', svm_evaluation, predictions, test_labels)
