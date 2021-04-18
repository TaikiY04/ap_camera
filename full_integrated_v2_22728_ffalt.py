# /etc/init.d/sample.py
### BEGIN INIT INFO
# Provides:          sample.py
# Required-Start:    $remote_fs $syslog
# Required-Stop:     $remote_fs $syslog
# Default-Start:     2 3 4 5
# Default-Stop:      0 1 6
# Short-Description: Start daemon at boot time
# Description:       Enable service provided by daemon.
### END INIT INFO

# import relevant libraries
import cv2
import sys
import numpy as np
import pigpio
import time
from time import sleep
from math import sin, cos, radians

##x-range: 700 to 2300
##y-range: 1400 to 2000

pi = pigpio.pi()
x_deg_eyes = 1500
x_deg_prof = 1500
x_deg_face = 1500
y_deg_face = 1700
pi.set_servo_pulsewidth(13, x_deg_face)
pi.set_servo_pulsewidth(26, y_deg_face)
counter = 1

############################ Set up our cascades ############################
# Open filepaths for cascades
cascPath = '/home/pi/Downloads/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml'
cascade2 = cv2.CascadeClassifier(cascPath)

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
    'minNeighbors': 3, 
    'minSize': (50, 50)
    }
settings_2 = {      #Profile face
    'scaleFactor': 1.1, 
    'minNeighbors': 20, 
    'minSize': (30, 30)
    }
settings_3 = {      #Eyes
    'scaleFactor': 20, 
    'minNeighbors': 3,
    'minSize': (1, 1),
    'maxSize': (20, 20)
    }

############################ Define our helper functions ############################
#def rotate_image(image, angle):
#    if angle == 0: return image
#    height, width = image.shape[:2]
#    rot_mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 0.9)
#    result = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
#    return result

#def rotate_point(pos, frame, angle):
#    if angle == 0: return pos
#    x = pos[0] - frame.shape[1]*0.4
#    y = pos[1] - frame.shape[0]*0.4
#    newx = x*cos(radians(angle)) + y*sin(radians(angle)) + frame.shape[1]*0.4
#    newy = -x*sin(radians(angle)) + y*cos(radians(angle)) + frame.shape[0]*0.4
#    return int(newx), int(newy), pos[2], pos[3]

############################ Main loop ############################
#Center:  320 x 240
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
x_ = int(240)
y_ = int(180)
w_ = int(640-2*(x_))
h_ = int(480-2*(y_))
downtime = 0
x_delta = 30
y_delta = 15
factor = 0.3
while True:
    #withinx = False
    #withiny = False
    # Capture frame-by-frame and convert to grayscale
    ret, original_frame = video_capture.read()
    frame = cv2.resize(original_frame, (0, 0), fx=factor, fy=factor)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(original_frame, (x_, y_), (x_+w_, y_+h_), clrs_6, 2)
    #t1 = time.perf_counter()
    default_face_alt = cascade2.detectMultiScale(gray, **settings_2)
    print(len(default_face_alt))

    if len(default_face_alt) != 0:
        downtime = 0
        #t2 = time.perf_counter()
        #print(f'{counter}: {1/(t2-t1):.2f} Hz')
        #counter = counter + 1
        #Eye detect
        #eyes = cascade7.detectMultiScale(gray, **settings_1)
        #for (x, y, w, h) in eyes[-1:]:
            #x *= 2
            #y *= 2
            #w *= 2
            #h *= 2
            #cv2.rectangle(original_frame, (x, y), (x+w, y+h), clrs_7, 2)
            #withinx = x_ <= x and x+w <= x_+w_ 
            #withiny = y_ <= y and y+h <= y_+h_
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
        #for angle in [0, -25, 25]:
        #    rimg = rotate_image(frame, angle)
        #    default_face_alt = cascade2.detectMultiScale(rimg, **settings_2)
        #    if len(default_face_alt):
        #        default_face_alt = [rotate_point(default_face_alt[-1], frame, -angle)]
        #        break
            #else:
            #    profile = cascade5.detectMultiScale(rimg, **settings_3)
            #    if len(profile):
            #        profile = [rotate_point(profile[-1], frame, -angle)]
            #for (x, y, w, h) in profile[-1:]:
            #    x *= 2
            #    y *= 2
            #   w *= 2
            #    h *= 2
            #    cv2.rectangle(original_frame, (x, y), (x+w, y+h), clrs_5, 2)
            #    withinx = x_ <= x and x+w <= x_+w_ 
            #    withiny = y_ <= y and y+h <= y_+h_
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
                    
        for (x, y, w, h) in default_face_alt[-1:]:
            x *= int(1/factor)
            y *= int(1/factor)
            w *= int(1/factor)
            h *= int(1/factor)
            cv2.rectangle(original_frame, (x, y), (x+w, y+h), clrs_1, 2)
            #withinx = x_ <= x and x+w <= x_+w_ 
            #withiny = y_ <= y and y+h <= y_+h_
            if (x_ + w_) <= (x + 2*w/3): #further (to the left of user)
                x_diff_face = (x + 2*w/3) - 320
                deg_change_face_x = 0.2*x_diff_face
                x_deg_face -= deg_change_face_x
                pi.set_servo_pulsewidth(13, x_deg_face)
            elif x + w/3 <= x_: #closer (to the right of user)
                x_diff_face = 320 - (x + w/3) 
                deg_change_face_x = 0.2*x_diff_face
                x_deg_face += deg_change_face_x
                pi.set_servo_pulsewidth(13, x_deg_face)
            if (y_ + h_) <= (y + 2*h/3): #bottom (below the user)
                y_diff_face = (y + 2*h/3) - 240
                deg_change_face_y = 0.3*y_diff_face
                y_deg_face -= deg_change_face_y
                pi.set_servo_pulsewidth(26, y_deg_face)
            elif y + h/3 <= y_: #top (below user)
                y_diff_face = 240 - (y + h/3) 
                deg_change_face_y = 0.3*y_diff_face
                y_deg_face += deg_change_face_y
                pi.set_servo_pulsewidth(26, y_deg_face)
            #if withinx and withiny:
            #    cv2.rectangle(original_frame, (x_, y_), (x_+w_, y_+h_), clrs_2, 2)
            #else:
            #    cv2.rectangle(original_frame, (x_, y_), (x_+w_, y_+h_), clrs_3, 2)
    else:
        downtime += 1
        if downtime >= 100:
            if x_deg_face > 2300:
                x_deg_face -= (x_deg_face - 2290)
                pi.set_servo_pulsewidth(13, x_deg_face)
                x_delta = -30
            elif x_deg_face < 700:
                x_deg_face += (710 - x_deg_face)
                pi.set_servo_pulsewidth(13, x_deg_face)
                x_delta = 30
            else:
                x_deg_face += x_delta
                pi.set_servo_pulsewidth(13, x_deg_face)
            if y_deg_face > 2000:
                y_deg_face -= (y_deg_face - 1990)
                pi.set_servo_pulsewidth(26, y_deg_face)
                y_delta = -30
            elif y_deg_face < 1400:
                y_deg_face += (1410 - y_deg_face)
                pi.set_servo_pulsewidth(26, y_deg_face)
                y_delta = 30
            else:
                y_deg_face += y_delta
                pi.set_servo_pulsewidth(26, y_deg_face)
            ret, original_frame = video_capture.read()
            frame = cv2.resize(original_frame, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.rectangle(original_frame, (x_, y_), (x_+w_, y_+h_), clrs_6, 2)
            default_face_alt = cascade2.detectMultiScale(gray, **settings_2)
            if len(default_face_alt):
                downtime = 0
                for (x, y, w, h) in default_face_alt[-1:]:
                    x *= int(1/factor)
                    y *= int(1/factor)
                    w *= int(1/factor)
                    h *= int(1/factor)
                    cv2.rectangle(original_frame, (x, y), (x+w, y+h), clrs_1, 2)
                
        # Display the resulting frame and any detected features
    cv2.imshow('Video', original_frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
        #print(x,y,w,h)
video_capture.release()
cv2.destroyAllWindows()
