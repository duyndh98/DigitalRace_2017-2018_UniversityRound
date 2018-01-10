from header import *

def get_red_mask(_hsv):  
    mask1 = cv2.inRange(_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(_hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    return mask

def get_blue_mask(_hsv):
    mask = cv2.inRange(_hsv, lower_blue, upper_blue)
    return mask

def get_mask(_hsv):
    red_mask = get_red_mask(_hsv)
    blue_mask = get_blue_mask(_hsv)
    return cv2.bitwise_or(red_mask, blue_mask)

def find_contour(_img, _mask):
    _mask, contours, hierarchy = cv2.findContours(_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(_img, contours, -1, (0, 255, 0), 1)
    
    height, width, channel = _img.shape
    max_area = 0
    max_bound = []
    
    for contour in contours:
        [x, y, w, h] = bound = cv2.boundingRect(contour)
        
        if (0.6 < w / h < 1.4) and (w > 20) and (h > 20):
            if w * h > max_area:
                max_area = w * h
                max_bound = bound                
    return max_bound

def ycrcb_equalize(_img):
    ycrcb = cv2.cvtColor(_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y = clahe.apply(y)

    ycrcb = cv2.merge([y, cr, cb])
    _img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def classify_image(_image, _svm):
    descriptor = hog.compute(_image)
    return int(_svm.predict(np.array([descriptor]))[1][0])

def process_image(_img, _svm, _templates, _templates_title, _file, _frame_id):
    ycrcb_equalize(_img)
    hsv = cv2.cvtColor(_img, cv2.COLOR_BGR2HSV)
    mask = get_mask(hsv)
    
    contour = find_contour(_img, mask)
    if contour:
        [x, y, w, h] = contour
        
        traffic_sign = _img[y:(y + h), x:(x + w)]
        traffic_sign = cv2.resize(traffic_sign, (width, height))
        class_id = classify_image(traffic_sign, _svm)
        '''
        # show template sign on the down-right of sign detected
        if (x + w + width < _img.shape[1]) and (y + height < _img.shape[0]):
            _img[y:(y + height), (x + w):(x + w + width)] = _templates[class_id - 1]
        # show template sign on the up-right of sign detected
        elif x + w + width < _img.shape[1]:
            _img[(y - height + h):(y+h), (x + w):(x + w + width)] = _templates[class_id - 1]
        # show template sign on the down-left of sign detected
        elif y + height < _img.shape[0]:
            _img[y:(y + height), (x - width):x] = _templates[class_id - 1]
        # show template sign on the up-left of sign detected
        else:
            _img[(y - height + h):(y+h), (x - width):x] = _templates[class_id - 1]
        '''
        # draw contour
        cv2.rectangle(_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        
        # show traffic sign infomation
        cv2.putText(_img, _templates_title[class_id - 1], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # save the traffic sign info to file
        _file.write(' '.join(str(x) for x in [_frame_id, class_id, x, y, x + w, y + w, '\n']))
    
    cv2.imshow('mask', mask)
    cv2.imshow('result', _img)