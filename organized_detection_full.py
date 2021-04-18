# import relevant libraries
import cv2
import sys
import numpy as np
import pigpio
from time import sleep
from math import sin, cos, radians

pi = pigpio.pi()
x_deg_eyes = 1500
x_deg_prof = 1500
x_deg_face = 1500
y_deg_face = 1600
pi.set_servo_pulsewidth(13, x_deg_face)
pi.set_servo_pulsewidth(26, y_deg_face)

############################ Set up our cascades ############################
# Open filepaths for cascades
classifier_pth = '/home/pi/Downloads/opencv-master/data/haarcascades/'

cascade1 = cv2.CascadeClassifier(''.join([classifier_pth, 'haarcascade_frontalface_default.xml']))
cascade2 = cv2.CascadeClassifier(''.join([classifier_pth, 'haarcascade_frontalface_alt.xml']))
cascade3 = cv2.CascadeClassifier(''.join([classifier_pth, 'haarcascade_frontalface_alt2.xml']))
cascade4 = cv2.CascadeClassifier(''.join([classifier_pth, 'haarcascade_frontalface_alt_tree.xml']))
cascade5 = cv2.CascadeClassifier(''.join([classifier_pth, 'haarcascade_profileface.xml']))
cascade6 = cv2.CascadeClassifier(''.join([classifier_pth, 'haarcascade_upperbody.xml']))
cascade7 = cv2.CascadeClassifier(''.join([classifier_pth, 'haarcascade_eye.xml']))

# Set colors for each cascade
clrs_1 = (255, 0, 0)        # Blue
clrs_2 = (0, 255, 0)        # Green
clrs_3 = (0, 0, 255)        # Red
clrs_4 = (0, 0, 0)          # Black
clrs_5 = (255, 255, 0)      # Teal
clrs_6 = (0, 255, 255)      # Orange
clrs_7 = (255, 255, 255)    # White

# Set custom processing settings for each cascade
settings_1 = {      #Default face
    'scaleFactor': 1.3, 
    'minNeighbors': 5, 
    'minSize': (50, 50)
    }
settings_2 = {      #Profile face
    'scaleFactor': 1.3, 
    'minNeighbors': 5, 
    'minSize': (50, 50)
    }
settings_3 = {      #Eyes
    'scaleFactor': 20, 
    'minNeighbors': 3,
    'minSize': (1, 1),
    'maxSize': (20, 20)
    }

############################ Define our helper functions ############################
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

############################ Do the main loop ############################
#Center:  320 x 240
video_capture = cv2.VideoCapture(0)
x_ = int(170)
y_ = int(150)
w_ = int(640-2*(x_))
h_ = int(480-2*(y_))
while True:
    withinx = False
    withiny = False
    # Capture frame-by-frame and convert to grayscale
    ret, original_frame = video_capture.read()
    frame = cv2.resize(original_frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(original_frame, (x_, y_), (x_+w_, y_+h_), clrs_6, 2)
    #Eye detect
    eyes = cascade7.detectMultiScale(gray, **settings_1)
    for (x, y, w, h) in eyes[-1:]:
        x *= 2
        y *= 2
        w *= 2
        h *= 2
        cv2.rectangle(original_frame, (x, y), (x+w, y+h), clrs_7, 2)
        withinx = x_ <= x and x+w <= x_+w_ 
        withiny = y_ <= y and y+h <= y_+h_
        #if (x_ + w_) <= (x + w/2):
        #    x_diff_eyes = (x + w/2) - 320
        #    deg_change_eyes = 0.5*x_diff_eyes
        #    while deg_change_eyes > 0:
        #        x_deg_eyes -= 20
        #        pi.set_servo_pulsewidth(13, x_deg_eyes)
        #        deg_change_eyes -= 20
        #        sleep(0.1)
        #elif x + w/2 <= x_:
        #    x_diff_eyes = 320 - (x + w/2) 
        #    deg_change_eyes = 0.5*x_diff_eyes
        #    while deg_change_eyes > 0:
        #        x_deg_eyes += 20
        #        pi.set_servo_pulsewidth(13, x_deg_eyes)
        #        deg_change_eyes -= 20
        #        sleep(0.1)
        #if withinx and withiny:
        #    cv2.rectangle(original_frame, (x_, y_), (x_+w_, y_+h_), clrs_2, 2)
        #else:
        #    cv2.rectangle(original_frame, (x_, y_), (x_+w_, y_+h_), clrs_3, 2)
    #Rotate Image
    for angle in [0, -25, 25]:
        rimg = rotate_image(frame, angle)
        default_face = cascade1.detectMultiScale(rimg, **settings_2)
        if len(default_face):
            default_face = [rotate_point(default_face[-1], frame, -angle)]
            break
        else:
            profile = cascade5.detectMultiScale(rimg, **settings_3)
            if len(profile):
                profile = [rotate_point(profile[-1], frame, -angle)]
        for (x, y, w, h) in profile[-1:]:
            x *= 2
            y *= 2
            w *= 2
            h *= 2
            cv2.rectangle(original_frame, (x, y), (x+w, y+h), clrs_5, 2)
            withinx = x_ <= x and x+w <= x_+w_ 
            withiny = y_ <= y and y+h <= y_+h_
            #if (x_ + w_) <= (x + w/2):
            #    x_diff_prof = (x + w/2) - 320
            #    deg_change_prof = 0.5*x_diff_prof
            #    while deg_change_prof > 0:
            #        x_deg_prof -= 20
            #        pi.set_servo_pulsewidth(13, x_deg_prof)
            #        deg_change_prof -= 20
            #        sleep(0.1)
            #elif x + w/2 <= x_:
            #    x_diff_prof = 320 - (x + w/2) 
            #    deg_change_prof = 0.5*x_diff_prof
            #    while deg_change_prof > 0:
            #        x_deg_prof += 20
            #        pi.set_servo_pulsewidth(13, x_deg_prof)
            #        deg_change_prof -= 20
            #        sleep(0.1)
            #if withinx and withiny:
            #    cv2.rectangle(original_frame, (x_, y_), (x_+w_, y_+h_), clrs_2, 2)
            #else:
            #    cv2.rectangle(original_frame, (x_, y_), (x_+w_, y_+h_), clrs_3, 2)
                
    for (x, y, w, h) in default_face[-1:]:
        x *= 2
        y *= 2
        w *= 2
        h *= 2
        cv2.rectangle(original_frame, (x, y), (x+w, y+h), clrs_1, 2)
        withinx = x_ <= x and x+w <= x_+w_ 
        withiny = y_ <= y and y+h <= y_+h_
        if (x_ + w_) <= (x + w/2):
            x_diff_face = (x + w/2) - 320
            deg_change_face_x = 0.4*x_diff_face
            while deg_change_face_x > 0:
                x_deg_face -= 30
                pi.set_servo_pulsewidth(13, x_deg_face)
                deg_change_face_x -= 30
                sleep(0.1)
        elif x + w/2 <= x_:
            x_diff_face = 320 - (x + w/2) 
            deg_change_face_x = 0.4*x_diff_face
            while deg_change_face_x > 0:
                x_deg_face += 30
                pi.set_servo_pulsewidth(13, x_deg_face)
                deg_change_face_x -= 30
                sleep(0.1)
        if (y_ + h_) <= (y + h/2):
            y_diff_face = (y + h/2) - 240
            deg_change_face_y = 0.4*y_diff_face
            while deg_change_face_y > 0:
                y_deg_face -= 20
                pi.set_servo_pulsewidth(26, y_deg_face)
                deg_change_face_y -= 20
                sleep(0.1)
        elif y + h/2 <= y_:
            y_diff_face = 240 - (y + h/2) 
            deg_change_face_y = 0.4*y_diff_face
            while deg_change_face_y > 0:
                y_deg_face += 20
                pi.set_servo_pulsewidth(26, y_deg_face)
                deg_change_face_y -= 20
                sleep(0.1)
        #if withinx and withiny:
        #    cv2.rectangle(original_frame, (x_, y_), (x_+w_, y_+h_), clrs_2, 2)
        #else:
        #    cv2.rectangle(original_frame, (x_, y_), (x_+w_, y_+h_), clrs_3, 2)

    # Display the resulting frame and any detected features
    cv2.imshow('Video', original_frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
        #print(x,y,w,h)
video_capture.release()
cv2.destroyAllWindows()
