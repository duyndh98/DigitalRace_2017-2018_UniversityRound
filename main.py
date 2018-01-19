
from header import *
from train import *
from image_processing import *

def machine_learning():
	print('*' * 20)
	print('MACHINE LEARNING')
	
	execute('=' * 20 + '\nTRAIN', train)
	
def process_video():
	print('*' * 20)
	print('PROCESS VIDEO')
	
	print('Init')
	svm = execute('Loading model', cv2.ml.SVM_load, svm_model_file)
	templates, templates_title = execute('Loading templates', load_templates)

	inp = cv2.VideoCapture(video_input)
	video_width = int(inp.get(cv2.CAP_PROP_FRAME_WIDTH))
	video_height = int(inp.get(cv2.CAP_PROP_FRAME_HEIGHT))
	video_fps = inp.get(cv2.CAP_PROP_FPS)
	
	print('Video resolution: (' + str(video_width) + ', ' + str(video_height) + ')')
	print('Video fps:', video_fps)

	out = cv2.VideoWriter(video_output, -1, video_fps, (normal_width, normal_height))
	
	print('Video is running')
	info = []
	
	frame_id = 1
	while inp.isOpened():
		ret, frame = inp.read()
		if (not ret) or (cv2.waitKey(1) & 0xFF == ord('q')):
			break
		frame = cv2.resize(frame, (normal_width, normal_height))
		process_image(frame, svm, templates, templates_title, frame_id, info)
		
		out.write(frame)
		frame_id += 1
        
	with open(output, 'w') as f:
		f.write(str(len(info)) + '\n')
		for elem in info:
		    f.write(' '.join(str(x) for x in elem))
		    
	inp.release()
	out.release()
	cv2.destroyAllWindows()

	
if __name__ == '__main__':
	# extract_video_datasets('D:\workspace\TrafficSignRecognitionAndDetection\Contest\datasets\Orginal\\abc')
	# create_train_datasets('D:\workspace\TrafficSignRecognitionAndDetection\Contest\datasets\Orginal\\abc\_10')
	# machine_learning()
	process_video()

	'''
	for img_dir in glob.glob('D:\workspace\\TrafficSignRecognitionAndDetection\Contest\datasets\Images\_5\*'):
		print(img_dir.split('.')[0] + '111' + '.jpg')
		img = cv2.imread(img_dir)
		img = cv2.flip(img, 1)
		cv2.imwrite(img_dir.split('.')[0] + '_flip' + '.jpg', img)
	'''