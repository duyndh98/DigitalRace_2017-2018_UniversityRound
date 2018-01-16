from header import *

def get_red_mask(_hsv):  
    return cv2.inRange(_hsv, lower_red1, upper_red1) | cv2.inRange(_hsv, lower_red2, upper_red2)

def get_blue_mask(_hsv):
    return cv2.inRange(_hsv, lower_blue, upper_blue)
    
def get_white_mask(_hsv):
    return cv2.inRange(_hsv, lower_white, upper_white)
    
def get_black_mask(_hsv):
    return cv2.inRange(_hsv, lower_black, upper_black)

def get_mask(_img):
    ycrcb_equalize(_img)
    hsv = cv2.cvtColor(_img, cv2.COLOR_BGR2HSV)
    mask = get_blue_mask(hsv) | get_red_mask(hsv) # get_white_mask(hsv) | get_black_mask(hsv)
    # kernel = np.ones((3, 3),np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations = 1)
    # mask = cv2.erode(mask, kernel, iterations = 1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)    
    return mask

def check_circles(_mask):
    if cv2.HoughCircles(cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, 1, minDist= min(width, height),param1=50,param2=30,minRadius=0,maxRadius=min(width, height)) == []:
        return True
    else:
        return False

def find_contour(_img, _mask):
    _mask, contours, hierarchy = cv2.findContours(_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(_img, contours, -1, (0, 255, 0), 1)
    
    height, width, channel = _img.shape
    max_area = 0
    max_bound = []
    
    for contour in contours:
        [x, y, w, h] = bound = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)
        ellipse_area = (math.pi * (w / 2) * (h / 2))

        if (0.5 < w / h < 1.5) and (w > 20) and (h > 20):
            if (0.6 < (contour_area / ellipse_area) < 1.4): # and check_circles(_mask):
                if w * h > max_area:
                    max_area = w * h
                    max_bound = bound
    return max_bound

def ycrcb_equalize(_img):
    ycrcb = cv2.cvtColor(_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4,4))
    y = clahe.apply(y)

    ycrcb = cv2.merge([y, cr, cb])
    _img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return _img

def classify_image(_image, _svm):
    descriptor = hog.compute(_image)
    return int(_svm.predict(np.array([descriptor]))[1][0])

def show_template(_img, _x, _y, _w, _h, _class_id, _templates):
    # show template sign on the down-right of sign detected
    if (_x + _w + width < _img.shape[1]) and (_y + height < _img.shape[0]):
        _img[_y:(_y + height), (_x + _w):(_x + _w + width)] = _templates[_class_id - 1]
    # show template sign on the up-right of sign detected
    elif _x + _w + width < _img.shape[1]:
        _img[(_y - height + h):(_y+h), (_x + _w):(_x + _w + width)] = _templates[_class_id - 1]
    # show template sign on the down-left of sign detected
    elif _y + height < _img.shape[0]:
        _img[_y:(_y + height), (_x - width):_x] = _templates[_class_id - 1]
    # show template sign on the up-left of sign detected
    else:
        _img[(_y - height + _h):(_y+_h), (_x - width):_x] = _templates[_class_id - 1]

def process_image(_img, _svm, _templates, _templates_title, _file, _frame_id):
    img = ycrcb_equalize(_img)
    mask = get_mask(img)
    
    contour = find_contour(img, mask)
    if contour:
        [x, y, w, h] = contour
        
        traffic_sign = img[y:(y + h), x:(x + w)]
        traffic_sign = cv2.resize(traffic_sign, (width, height))
        class_id = classify_image(traffic_sign, _svm)
        
        if class_id != 11:
            # draw contour
            cv2.rectangle(_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            
            # show template of traffic sign
            show_template(_img, x, y, w, h, class_id, _templates)

            # show traffic sign infomation
            cv2.putText(_img, _templates_title[class_id - 1], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            # save the traffic sign info to file
            _file.write(' '.join(str(x) for x in [_frame_id, class_id, x, y, x + w, y + w, '\n']))
    
    cv2.imshow('mask', mask)
    cv2.imshow('result', _img)