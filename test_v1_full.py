# import relevant libraries
import cv2
import sys
import numpy as np
from math import sin, cos, radians

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
clrs_in = (0, 255, 0)
clrs_out = (0, 0, 255)

# Set custom processing settings for each cascade
settings_1 = {
    'scaleFactor': 1.3, 
    'minNeighbors': 3, 
    'minSize': (50, 50)
    }
settings_2 = {
    'scaleFactor': 1.3, 
    'minNeighbors': 3, 
    'minSize': (50, 50)
    }
settings_3 = {
    'scaleFactor': 1.3, 
    'minNeighbors': 3, 
    'minSize': (50, 50)
    }
x_1, y_1, = 220, 120

w_1, h_1 = (640-2*(x_1)), (480-2*(y_1))

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
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame and convert to grayscale
    ret, original_frame = video_capture.read()
    frame = cv2.resize(original_frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(original_frame, (x_1, y_1), ((x_1+w_1), (y_1 + h_1)), clrs_out, 2)
    
    #Eye detect
    eyes = cascade7.detectMultiScale(gray, **settings_1)
    for (x, y, w, h) in eyes[-1:]:
            x *= 2
            y *= 2
            w *= 2
            h *= 2
            cv2.rectangle(original_frame, (x, y), (x+w, y+h), clrs_7, 2)
            if x_1 <= (x + w/2) <= (x_1+w_1) and y_1 <= (y + h/2) <= (y_1+h_1):
                #print("inside")
                cv2.rectangle(original_frame, (x_1, y_1), ((x_1+w_1), (y_1 + h_1)), clrs_in, 2)
            else:
                #print("outside")
                cv2.rectangle(original_frame, (x_1, y_1), ((x_1+w_1), (y_1 + h_1)), clrs_out, 2)

    #Rotate Image
    for angle in [0, -25, 25]:
        rimg = rotate_image(frame, angle)
        default_face = cascade1.detectMultiScale(rimg, **settings_2)
        profile = cascade5.detectMultiScale(rimg, **settings_3)
        if len(default_face):
            default_face = [rotate_point(default_face[-1], frame, -angle)]
            break
        else:
            if len(profile):
                profile = [rotate_point(profile[-1], frame, -angle)]
        for (x, y, w, h) in profile[-1:]:
            x *= 2
            y *= 2
            w *= 2
            h *= 2
            cv2.rectangle(original_frame, (x, y), (x+w, y+h), clrs_5, 2)
            if x_1 <= (x + w/2) <= (x_1+w_1) and y_1 <= (y + h/2) <= (y_1+h_1):
                #print("inside")
                cv2.rectangle(original_frame, (x_1, y_1), ((x_1+w_1), (y_1 + h_1)), clrs_in, 2)
            else:
                #print("outside")
                cv2.rectangle(original_frame, (x_1, y_1), ((x_1+w_1), (y_1 + h_1)), clrs_out, 2)

    for (x, y, w, h) in default_face[-1:]:
            x *= 2
            y *= 2
            w *= 2
            h *= 2
            cv2.rectangle(original_frame, (x, y), (x+w, y+h), clrs_1, 2)
            if x_1 <= (x + w/2) <= (x_1+w_1) and y_1 <= (y + h/2) <= (y_1+h_1):
                #print("inside")
                cv2.rectangle(original_frame, (x_1, y_1), ((x_1+w_1), (y_1 + h_1)), clrs_in, 2)
            else:
                #print("outside")
                cv2.rectangle(original_frame, (x_1, y_1), ((x_1+w_1), (y_1 + h_1)), clrs_out, 2)
        
    # Display the resulting frame and any detected features
    cv2.imshow('Video', original_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #if len(default_face):
        #print (x, y, w, h)
    #if len(profile):
        #print (x, y, w, h)
    print(np.shape(original_frame))

video_capture.release()
cv2.destroyAllWindows()
