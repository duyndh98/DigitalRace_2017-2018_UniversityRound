from header import *

def get_mask(_img, _colors):
    # _img = cv2.GaussianBlur(_img, (3, 3), 0)
    ycrcb_equalize(_img)
    hsv = cv2.cvtColor(_img, cv2.COLOR_BGR2HSV)

    mask = np.zeros(_img.shape[:2],np.uint8)
    if 'blue' in _colors:
        mask = mask | cv2.inRange(hsv, lower_blue, upper_blue)
    if 'red' in _colors:
        mask = mask | cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    if 'white' in _colors:
        mask = mask | cv2.inRange(hsv, lower_white, upper_white)
    if 'black' in _colors:
        mask = mask | cv2.inRange   (hsv, lower_black, upper_black)
    
    kernel = np.ones((5, 5),np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    mask = cv2.erode(mask, kernel, iterations = 1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)    
    return mask

def check_is_circles(_img):
    gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
    return (cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 
        minDist= gray.shape[0], param1=50, param2=30,
        minRadius=int(gray.shape[1]/2-5),maxRadius=int(gray.shape[0]/2+5)) is not None)

def find_bounds(_mask, _img):
    _mask, contours, hierarchy = cv2.findContours(_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounds = []
    for contour in contours:
        [x, y, w, h] = bound = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)
        ellipse_area = (math.pi * (w / 2) * (h / 2))

        if (0.4 < w / h < 1.6) and (w > 20) and (h > 20):
            if 0.8 < (contour_area / ellipse_area) < 1.2:
                if True:#check_is_circles(_img[y:y+h, x:x+w]):
                    bounds.append(bound)

    return sorted(bounds, key=lambda x: (x[2] * x[3]))

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

def show_template(_img, _bound, _template):
    [x, y, w, h] = _bound
    _template = cv2.resize(_template, (h, h))
    # right
    if x + w + h < _img.shape[1]:
        _img[y : y + h, x + w: x + w + h] = _template
    # left
    else:
        _img[y : y + h, x - h: x] = _template
    
def regconize_sign(_img, _mask, _svm, _id):
    bounds = find_bounds(_mask, _img)
    
    for bound in bounds:
        [x, y, w, h] = bound
        
        sign = _img[y:(y + h), x:(x + w)]
        sign = cv2.resize(sign, (width, height))
        class_id = classify_image(sign, _svm)
        
        if class_id == _id:
            return bound, class_id
    return [], 11

def draw(_img, _bound, _template, _template_title):
    [x, y, w, h] = _bound
    # draw contour
    cv2.rectangle(_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    
    # show template of traffic sign
    show_template(_img, _bound, _template)
    # show traffic sign infomation
    # cv2.putText(_img, _templates_title, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    return _img

def process_image(_img, _svm, _templates, _templates_title, _frame_id, _info):
    img = ycrcb_equalize(_img)
    '''
    blue_mask = get_mask(img, ['blue'])
    red_mask = get_mask(img, ['red'])
    blue_red_mask = get_mask(img, ['blue', 'red'])
    white_black_mask = get_mask(img, ['white', 'black'])

    blue_bound, blue_predict_id = regconize_sign(img, blue_mask, _svm)
    red_bound, red_predict_id = regconize_sign(img, red_mask, _svm)
    blue_red_bound, blue_red_predict_id = regconize_sign(img, blue_red_mask, _svm)
    white_black_bound, white_black_predict_id = regconize_sign(img, white_black_mask, _svm)
    ''' 
    mask = get_mask(img, ['blue'])
    wanted_id = 8
    templates_id = blue_templates_id
    
    bound, predict_id = regconize_sign(img, mask, _svm, wanted_id)
    if predict_id == wanted_id:
        draw(_img, bound, _templates[predict_id - 1], _templates_title[predict_id - 1])
        [x, y, w, h] = bound
        _info.append([_frame_id, predict_id, x, y, x + w, y + h, '\n'])
        
    # cv2.imshow('mask', mask)
    cv2.imshow('result', _img)
    cv2.imshow('mask', mask)
    # cv2.imshow('blue_mask', blue_mask)
    # cv2.imshow('red_mask', red_mask)
    # cv2.imshow('white_black_mask', white_black_mask)