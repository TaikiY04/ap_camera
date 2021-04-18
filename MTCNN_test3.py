from mtcnn.mtcnn import MTCNN
import cv2
import sys
import numpy as np
import pigpio
import time
from time import sleep
from math import sin, cos, radians

detector = MTCNN()
counter = 0
timer = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#python3 /home/pi/Downloads/opencv_object_tracking.py --tracker csrt

# Set colors for each cascade
clrs_1 = (255, 0, 0)        # Blue
clrs_2 = (0, 255, 0)        # Green
clrs_3 = (0, 0, 255)        # Red
clrs_4 = (0, 0, 0)          # Black
clrs_5 = (255, 255, 0)      # Teal
clrs_6 = (0, 255, 255)      # Orange
clrs_7 = (255, 255, 255)    # White

pi = pigpio.pi()
x_deg_eyes = 1500
x_deg_prof = 1500
x_deg_face = 1500
y_deg_face = 1700
pi.set_servo_pulsewidth(13, x_deg_face)
pi.set_servo_pulsewidth(26, y_deg_face)

factor = 0.4
x_ = int(210*factor)
y_ = int(170*factor)
w_ = int(640*factor-2*(x_))
h_ = int(480*factor-2*(y_))
downtime = 0                                                                                                                                                                                                                                                 
x_delta = 30
y_delta = 15

while True: 
    #Capture frame-by-frame
    __, frame = cap.read()
    frame = cv2.resize(frame, (0,0), fx= factor, fy= factor)
    cv2.rectangle(frame, (x_, y_), (x_+w_, y_+h_), clrs_6, 2)
    
    #Use MTCNN to detect faces
    t1 = time.perf_counter()
    result = detector.detect_faces(frame)
    t2 = time.perf_counter()
    print(f'{counter}: {1/(t2-t1):.2f} Hz')
    
    if result != []:
        for face in result:
            t3 = time.perf_counter()
            bbox = face['box']
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,155,255), 2)
            print(f'{bbox[2]} x {bbox[3]}')
            if (x_ + w_) <= (x + 2*w/3): #further (to the left of user)
                x_diff_face = ((x/factor) + 2*(w/factor)/3) - 320
                deg_change_face_x = 0.3*x_diff_face
                x_deg_face -= deg_change_face_x
                pi.set_servo_pulsewidth(13, x_deg_face)
            elif x + w/3 <= x_: #closer (to the right of user)
                x_diff_face = 320 - ((x/factor) + (w/factor)/3) 
                deg_change_face_x = 0.3*x_diff_face
                x_deg_face += deg_change_face_x
                pi.set_servo_pulsewidth(13, x_deg_face)
            if (y_ + h_) <= (y + 2*h/3): #bottom (below the user)
                y_diff_face = ((y/factor) + 2*(h/factor)/3) - 240
                deg_change_face_y = 0.4*y_diff_face
                y_deg_face -= deg_change_face_y
                pi.set_servo_pulsewidth(26, y_deg_face)
            elif y + h/3 <= y_: #top (below user)
                y_diff_face = 240 - ((y/factor) + (h/factor)/3) 
                deg_change_face_y = 0.4*y_diff_face
                y_deg_face += deg_change_face_y
                pi.set_servo_pulsewidth(26, y_deg_face)
            t4 = time.perf_counter()
            print(f'{timer}: {1/(t4 - t3):.2f} Hz')
    
    counter = counter + 1
    timer = counter
    #display resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
        
#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()
