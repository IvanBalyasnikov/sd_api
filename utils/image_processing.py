import numpy as np
import base64
import cv2

def cv_b64_im(img):
    retval, bytes = cv2.imencode('.png', img)
    return base64.b64encode(bytes).decode('utf-8')


def b64_cv_im(img_str):
    im_bytes = base64.b64decode(img_str)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  
    return cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
