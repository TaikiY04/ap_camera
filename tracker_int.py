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

factor = 0.5
moby_factor = 0.7
x_ = int(220*factor)
y_ = int(170*factor)
w_ = int(640*factor-2*(x_))
h_ = int(480*factor-2*(y_))

#def servo_move(x, y, w, h, position_x, position_y):
##    if (x_ + w_) <= (x + 2*w/3):
##        x_diff_face = ((x/factor) + 2*(w/factor)/3) - 320
##        deg_change_face_x = 0.3*x_diff_face
##        position_x -= deg_change_face_x
##        pi.set_servo_pulsewidth(13, x_deg_face)
##    elif x + w/3 <= x_:
##        x_diff_face = 320 - ((x/factor) + (w/factor)/3) 
##        deg_change_face_x = 0.3*x_diff_face
##        position_x += deg_change_face_x
##        pi.set_servo_pulsewidth(13, x_deg_face)
##    if (y_ + h_) <= (y + 2*h/3):
##        y_diff_face = ((y/factor) + 2*(h/factor)/3) - 240
##        deg_change_face_y = 0.4*y_diff_face
##        position_y -= deg_change_face_y
##        pi.set_servo_pulsewidth(26, y_deg_face)
##    elif y + h/3 <= y_:
##        y_diff_face = 240 - ((y/factor) + (h/factor)/3) 
##        deg_change_face_y = 0.4*y_diff_face
##        position_y += deg_change_face_y
##        pi.set_servo_pulsewidth(26, y_deg_face)


OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create, #3.4 fps
        "kcf": cv2.TrackerKCF_create, #6.5 fps
        "boosting": cv2.TrackerBoosting_create, #4.8 fps
        "mil": cv2.TrackerMIL_create, #3.4 fps
        "tld": cv2.TrackerTLD_create, #4.3 fps, jumpy af
        "medianflow": cv2.TrackerMedianFlow_create, #10.3, switchs often, might be kinky
        "mosse": cv2.TrackerMOSSE_create #10 fps, prolly the best one
    }

tracker = OPENCV_OBJECT_TRACKERS["csrt"]()
counter = 0
nuttime = 0
t1 = 0
t2 = 0
default = 0

while True: 
    __, moby_frame = cap.read()
    t3 = time.perf_counter()
    bigger_frame = cv2.resize(moby_frame, (0,0), fx= moby_factor, fy= moby_factor)
    frame = cv2.resize(moby_frame, (0,0), fx= factor, fy= factor)
    cv2.rectangle(frame, (x_, y_), (x_+w_, y_+h_), clrs_6, 2)
    t2 = time.perf_counter()
    nuttime = t2 - t1
    if default == 0 or nuttime >= 0.5:
        result = detector.detect_faces(bigger_frame)
        
        if result != []:
            for face in result:
                bbox = face['box']
                x = int(bbox[0]*(factor/moby_factor))
                y = int(bbox[1]*(factor/moby_factor))
                w = int(bbox[2]*(factor/moby_factor))
                h = int(bbox[3]*(factor/moby_factor))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,155,255), 2)
                #servo_move(x, y, w, h, x_deg_face, y_deg_face)
##                if (x_ + w_) <= (x + 2*w/3):
##                    x_diff_face = ((x/factor) + 2*(w/factor)/3) - 320
##                    deg_change_face_x = 0.3*x_diff_face
##                    x_deg_face -= deg_change_face_x
##                    pi.set_servo_pulsewidth(13, x_deg_face)
##                elif x + w/3 <= x_:
##                    x_diff_face = 320 - ((x/factor) + (w/factor)/3) 
##                    deg_change_face_x = 0.3*x_diff_face
##                    x_deg_face += deg_change_face_x
##                    pi.set_servo_pulsewidth(13, x_deg_face)
##                if (y_ + h_) <= (y + 2*h/3):
##                    y_diff_face = ((y/factor) + 2*(h/factor)/3) - 240
##                    deg_change_face_y = 0.4*y_diff_face
##                    y_deg_face -= deg_change_face_y
##                    pi.set_servo_pulsewidth(26, y_deg_face)
##                elif y + h/3 <= y_:
##                    y_diff_face = 240 - ((y/factor) + (h/factor)/3) 
##                    deg_change_face_y = 0.4*y_diff_face
##                    y_deg_face += deg_change_face_y
##                    pi.set_servo_pulsewidth(26, y_deg_face)
                if x != None and default == 1:
                    tracker = None
                tracker = OPENCV_OBJECT_TRACKERS["csrt"]()
                tracker.init(frame, (x,y,w,h))
                t1 = time.perf_counter()
                default = 1
    
    else:
        (success, box) = tracker.update(frame)
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #servo_move(x, y, w, h, x_deg_face, y_deg_face)
        if (x_ + w_) <= (x + 2*w/3):
            x_diff_face = ((x/factor) + 2*(w/factor)/3) - 320
            deg_change_face_x = 0.2*x_diff_face
            x_deg_face -= deg_change_face_x
            pi.set_servo_pulsewidth(13, x_deg_face)
        elif x + w/3 <= x_:
            x_diff_face = 320 - ((x/factor) + (w/factor)/3) 
            deg_change_face_x = 0.2*x_diff_face
            x_deg_face += deg_change_face_x
            pi.set_servo_pulsewidth(13, x_deg_face)
        if (y_ + h_) <= (y + 2*h/3):
            y_diff_face = ((y/factor) + 2*(h/factor)/3) - 240
            deg_change_face_y = 0.3*y_diff_face
            y_deg_face -= deg_change_face_y
            pi.set_servo_pulsewidth(26, y_deg_face)
        elif y + h/3 <= y_:
            y_diff_face = 240 - ((y/factor) + (h/factor)/3) 
            deg_change_face_y = 0.3*y_diff_face
            y_deg_face += deg_change_face_y
            pi.set_servo_pulsewidth(26, y_deg_face)
    cv2.imshow('frame', frame)
    t4 = time.perf_counter()
    print(f'{counter}: {1/(t4-t3):.2f} Hz')
    if 1/(t4-t3) >= 15:
        x_deg_face = 1500
        y_deg_face = 1700
        pi.set_servo_pulsewidth(13, x_deg_face)
        pi.set_servo_pulsewidth(26, y_deg_face)
    
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
