from header import *

def get_mask(_img, _colors):
    _img = cv2.GaussianBlur(_img, (3, 3), 0)
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
        mask = mask | cv2.inRange(hsv, lower_black, upper_black)
    
    kernel = np.ones((3, 3),np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    # mask = cv2.erode(mask, kernel, iterations = 1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)    
    return mask

def find_bounds(_mask):
    _mask, contours, hierarchy = cv2.findContours(_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounds = []
    for contour in contours:
        [x, y, w, h] = bound = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)
        ellipse_area = (math.pi * (w / 2) * (h / 2))

        if (0.4 < w / h < 1.6) and (w > 20) and (h > 20):
            if 0.8 < (contour_area / ellipse_area) < 1.2:
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
    
def regconize_sign(_img, _mask, _svm):
    bounds = find_bounds(_mask)
    
    for bound in bounds:
        [x, y, w, h] = bound
        
        sign = _img[y:(y + h), x:(x + w)]
        sign = cv2.resize(sign, (width, height))
        class_id = classify_image(sign, _svm)
        
        if class_id != 11:
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
    
    blue_mask = get_mask(img, ['blue'])
    red_mask = get_mask(img, ['red'])
    blue_red_mask = get_mask(img, ['blue', 'red'])
    white_black_mask = get_mask(img, ['white', 'black'])

    blue_bound, blue_class_id = regconize_sign(img, blue_mask, _svm)
    red_bound, red_class_id = regconize_sign(img, red_mask, _svm)
    blue_red_bound, blue_red_class_id = regconize_sign(img, blue_red_mask, _svm)
    white_black_bound, white_black_class_id = regconize_sign(img, white_black_mask, _svm)
    
    max_bound = [0, 0, 0, 0]
    class_id = 11
    if blue_bound != []:
        if blue_class_id not in blue_template_id:
            blue_bound = []
        elif blue_bound[2] * blue_bound[3] > max_bound[2] * max_bound[3]:
            max_bound = blue_bound
            class_id = blue_class_id
            # draw(_img, blue_bound, _templates[blue_class_id - 1], _templates_title[blue_class_id - 1])
    if red_bound != []:
        if red_class_id not in red_template_id:
            red_bound = []
        elif red_bound[2] * red_bound[3] > max_bound[2] * max_bound[3]:
            max_bound = red_bound
            class_id = red_class_id
            # draw(_img, red_bound, _templates[red_class_id - 1], _templates_title[red_class_id - 1])
    if blue_red_bound != []:
        if blue_red_class_id not in blue_red_template_id:
            blue_red_bound = []
        elif blue_red_bound[2] * blue_red_bound[3] > max_bound[2] * max_bound[3]:
            max_bound = blue_red_bound
            class_id = blue_red_class_id
            # draw(_img, blue_red_bound, _templates[blue_red_class_id - 1], _templates_title[blue_red_class_id - 1])
    if white_black_bound != []:
        if white_black_class_id not in white_black_template_id:
            white_black_bound = []
        elif white_black_bound[2] * white_black_bound[3] > max_bound[2] * max_bound[3]:
            max_bound = white_black_bound
            class_id = white_black_class_id
            # draw(_img, white_black_bound, _templates[white_black_class_id - 1], _templates_title[white_black_class_id - 1])

    if max_bound != [0, 0, 0, 0]:
        draw(_img, max_bound, _templates[class_id - 1], _templates_title[class_id - 1])
    
    [x, y, w, h] = max_bound
    _info.append([_frame_id, class_id, x, y, x + w, y + h, '\n'])
    
    # cv2.imshow('mask', mask)
    cv2.imshow('result', _img)
    cv2.imshow('blue_mask', blue_mask)
    cv2.imshow('red_mask', red_mask)
    cv2.imshow('white_black_mask', white_black_mask)
