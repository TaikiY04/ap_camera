import cv2
import sys
import numpy as np
import pigpio
import time
from time import sleep
from math import sin, cos, radians

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
counter = 1

modelFile = "/home/pi/Downloads/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "/home/pi/Downloads/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
x_ = int(200)
y_ = int(150)
w_ = int(640-2*(x_))
h_ = int(480-2*(y_))
downtime = 0
x_delta = 30
y_delta = 15
counter = 1
frame_process = True

while True:
    ret, img = video_capture.read()
    #img2 = cv2.resize(img, (300, 300))
    img2 = img
    cv2.rectangle(img2, (x_, y_), (x_+w_, y_+h_), clrs_6, 2)

    if frame_process:
        t1 = time.perf_counter()
        blob = cv2.dnn.blobFromImage(img2, 1.0, (175, 175), (104.0, 117.0, 123.0))
        net.setInput(blob)
        faces = net.forward()
        t2 = time.perf_counter()
        print(f'{counter}: {1/(t2-t1):.2f} Hz')
        
        #to draw faces on image
        h, w = img2.shape[:2]
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.3:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(img2, (x, y), (x1, y1), (0, 0, 255), 2)
                print(f'{x1-x} x {y1-y} | {confidence:.2f}')
                w_b = x1
                h_b = y1
                if (x_ + w_) <= (x + 2*w_b/3): #further (to the left of user)
                    x_diff_face = (x + 2*w_b/3) - 320
                    deg_change_face_x = 0.2*x_diff_face
                    x_deg_face -= deg_change_face_x
                    pi.set_servo_pulsewidth(13, x_deg_face)
                elif x + w_b/3 <= x_: #closer (to the right of user)
                    x_diff_face = 320 - (x + w_b/3) 
                    deg_change_face_x = 0.2*x_diff_face
                    x_deg_face += deg_change_face_x
                    pi.set_servo_pulsewidth(13, x_deg_face)
                if (y_ + h_) <= (y + 2*h_b/3): #bottom (below the user)
                    y_diff_face = (y + 2*h_b/3) - 240
                    deg_change_face_y = 0.3*y_diff_face
                    y_deg_face -= deg_change_face_y
                    pi.set_servo_pulsewidth(26, y_deg_face)
                elif y + h_b/3 <= y_: #top (below user)
                    y_diff_face = 240 - (y + h_b/3) 
                    deg_change_face_y = 0.3*y_diff_face
                    y_deg_face += deg_change_face_y
                    pi.set_servo_pulsewidth(26, y_deg_face)
        counter = counter + 1
    frame_process = not frame_process
    cv2.imshow('Video', img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
video_capture.release()
cv2.destroyAllWindows()
