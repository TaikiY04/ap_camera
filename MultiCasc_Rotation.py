import cv2
import sys
import numpy as np
from math import sin, cos, radians

classifier_pth = 'C:/Users/taiya/Downloads/opencv-master/opencv-master/data/haarcascades/'

cascade1 = cv2.CascadeClassifier(''.join([classifier_pth, 'haarcascade_frontalface_default.xml']))
cascade2 = cv2.CascadeClassifier(''.join([classifier_pth, 'haarcascade_frontalface_alt.xml']))
cascade3 = cv2.CascadeClassifier(''.join([classifier_pth, 'haarcascade_frontalface_alt2.xml']))
cascade4 = cv2.CascadeClassifier(''.join([classifier_pth, 'haarcascade_frontalface_alt_tree.xml']))
cascade5 = cv2.CascadeClassifier(''.join([classifier_pth, 'haarcascade_profileface.xml']))
cascade6 = cv2.CascadeClassifier(''.join([classifier_pth, 'haarcascade_upperbody.xml']))
cascade7 = cv2.CascadeClassifier(''.join([classifier_pth, 'haarcascade_eye.xml']))

casc_list = [[cascade1, (255, 0, 0)],       # Blue
             [cascade2, (0, 255, 0)],       # Green
             [cascade3, (0, 0, 255)],       # Red
             [cascade4, (0, 0, 0)],         # Black
             [cascade5, (255, 255, 0)],     # Teal
             [cascade6, (0, 255, 255)],     # Orange
             [cascade7, (255, 255, 255)]    # White
            ]
casc_list = [[cascade1, (255, 0, 0)],   
             [cascade5, (255, 255, 0)],
             [cascade7, (255, 255, 255)]
            ]

def rotate_image(image, angle):
    if angle == 0: return image
    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 0.9)
    result = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return result

def rotate_point(pos, frame, angle):
    if angle == 0: return pos
    x = pos[0] - frame.shape[1]*0.4
    y = pos[1] - frame.shape[0]*0.4
    newx = x*cos(radians(angle)) + y*sin(radians(angle)) + frame.shape[1]*0.4
    newy = -x*sin(radians(angle)) + y*cos(radians(angle)) + frame.shape[0]*0.4
    return int(newx), int(newy), pos[2], pos[3]

def find_feature(cascade, box_clr, gray, frame):
    faces = {
        'scaleFactor':1.1,
        'minNeighbors':10,     # Lowering this makes the algo less sensitive and alow it to find more features
        'minSize':(30, 30)#,
        #flags = cv2.CASCADE_FIND_BIGGEST_OBJECT
    }
    
    for angle in [0, -25, 25]:
        rimg = rotate_image(frame, angle)
        detected = cascade.detectMultiScale(rimg, **faces)
        if len(detected):
            detected = [rotate_point(detected[-1], frame, -angle)]
            break
        
    # Draw a rectangle around the faces
    for (x, y, w, h) in detected[-1:]:
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_clr, 2)
        
    return frame

video_capture = cv2.VideoCapture(0)

while True:
    
    # Capture frame-by-frame and convert to grayscale
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Iterate through our cascade models
    for k in range(len(casc_list)):
        frame = find_feature(casc_list[k][0], casc_list[k][1], gray, frame)
    
    # Display the resulting frame and any detected features
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
