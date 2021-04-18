from mtcnn.mtcnn import MTCNN
import cv2
import sys
import numpy as np
import pigpio
import time
from time import sleep
from math import sin, cos, radians

detector = MTCNN()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

clrs_6 = (0, 255, 255)

pi = pigpio.pi()
x_deg_face = 1500
y_deg_face = 1700
pi.set_servo_pulsewidth(13, x_deg_face)
pi.set_servo_pulsewidth(26, y_deg_face)
factor = 0.4
center_x = factor*(640)/2
center_y = factor*(480)/2


x_ = int(220*factor)
y_ = int(170*factor)
w_ = int(640*factor-2*(x_))
h_ = int(480*factor-2*(y_))

while True: 
    __, frame = cap.read()
    frame = cv2.resize(frame, (0,0), fx= factor, fy= factor)
    cv2.rectangle(frame, (x_, y_), (x_+w_, y_+h_), clrs_6, 2)
    result = detector.detect_faces(frame)
    
    if result != []:
        for face in result:
            t3 = time.perf_counter()
            bbox = face['box']
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,155,255), 2)
            if (x_ + w_) <= (x + 2*w/3):
                x_diff_face = ((x/factor) + 2*(w/factor)/3) - center_x
                deg_change_face_x = 0.3*x_diff_face
                x_deg_face -= deg_change_face_x
                pi.set_servo_pulsewidth(13, x_deg_face)
            elif x + w/3 <= x_:
                x_diff_face = center_x - ((x/factor) + (w/factor)/3) 
                deg_change_face_x = 0.3*x_diff_face
                x_deg_face += deg_change_face_x
                pi.set_servo_pulsewidth(13, x_deg_face)
            if (y_ + h_) <= (y + 2*h/3):
                y_diff_face = ((y/factor) + 2*(h/factor)/3) - center_y
                deg_change_face_y = 0.4*y_diff_face
                y_deg_face -= deg_change_face_y
                pi.set_servo_pulsewidth(26, y_deg_face)
            elif y + h/3 <= y_:
                y_diff_face = center_y - ((y/factor) + (h/factor)/3) 
                deg_change_face_y = 0.4*y_diff_face
                y_deg_face += deg_change_face_y
                pi.set_servo_pulsewidth(26, y_deg_face)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
