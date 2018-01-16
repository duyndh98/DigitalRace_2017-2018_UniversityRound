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
	
	with open(output, 'w') as f:
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
		frame_id = 0
		while inp.isOpened():
			ret, frame = inp.read()
			if (not ret) or (cv2.waitKey(1) & 0xFF == ord('q')):
				break
			frame = cv2.resize(frame, (normal_width, normal_height))
			process_image(frame, svm, templates, templates_title, f, frame_id)
			
			out.write(frame)
			frame_id += 1
	        
		inp.release()
		out.release()
		cv2.destroyAllWindows()

if __name__ == '__main__':
	# extract_video_datasets('D:\workspace\TrafficSignRecognitionAndDetection\Contest\datasets\Orginal\\abc')
	# create_train_datasets('D:\workspace\TrafficSignRecognitionAndDetection\Contest\datasets\Orginal\\abc\_10')
	# machine_learning()
	process_video()
	