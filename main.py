from header import *
from train import *
from test import *

if __name__ == '__main__':	
	
	# After the training, you can skip this steps at the next time by loading svm_model file
	execute('=' * 20 + '\nTRAIN', train)
	
	execute('=' * 20 + '\nTEST', test)