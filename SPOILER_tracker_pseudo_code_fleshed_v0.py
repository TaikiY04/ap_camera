import cv2
import sys
import numpy as np
from math import sin, cos, radians
from time import sleep
############################ Set up our static variables ############################

# Open filepaths for cascades
classifier_pth = '/home/pi/Downloads/opencv-master/data/haarcascades/'
eyes_cascade = cv2.CascadeClassifier(''.join([classifier_pth, 'haarcascade_eye.xml']))
face_cascade = cv2.CascadeClassifier(''.join([classifier_pth, 'haarcascade_frontalface_default.xml']))
prof_cascade = cv2.CascadeClassifier(''.join([classifier_pth, 'haarcascade_profileface.xml']))

# Set colors for each cascade
eyes_clr = (255, 0, 0)
face_clr = (0, 0, 0)
prof_clr = (255, 255, 255)
box_line_width = 2 # Thickness for boxes that highlight our features

# Custom processing settings for each cascade
eyes_settings = {
    'scaleFactor': 20, 
    'minNeighbors': 3,  # Discriminate a bit more strongly on what is an eye
    'minSize': (10, 10) # Allow for eyes to be smaller than other features
    }

face_settings = {
    'scaleFactor': 1.1, 
    'minNeighbors': 3, 
    'minSize': (20, 20)
    }

prof_settings = {
    'scaleFactor': 1.1, 
    'minNeighbors': 5,  
    'minSize': (20, 20)
    }


# Other things
angs2scan = [25, -25]       # Angles we want to scan through if we don't find anything @ 0 degrees

# TODO: Stuff to highlight whether or not we're in/our of bounds
# bound_coords = [1,2,3,4]    # Some list of coordinates to define the limits for when we move the camera
# coord_list_len = 10         # Length of our list to store our coordinates; this will help prevent spurious movements
# in_clr  = (0, 255, 0)     
# out_clr = (0, 0, 255)    


############################ Define our helper functions ############################
# Function to rotate our images
def rotate_image(image, angle):
    if angle == 0: return image
    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 0.9)
    result = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return result

# Function to rotate any detected features back to original frame of reference
def rotate_point(pos, img, angle):
    if angle == 0: return pos
    
    # Iterate through all of the highlighted features and translate the coordinates
    for k in range( np.shape(pos)[0] ):
        
        x = pos[k,0] - img.shape[1]*0.4
        y = pos[k,1] - img.shape[0]*0.4
        newx = x*cos(radians(angle)) + y*sin(radians(angle)) + img.shape[1]*0.4
        newy = -x*sin(radians(angle)) + y*cos(radians(angle)) + img.shape[0]*0.4
    
        pos[k, :] = np.array([int(newx), int(newy), pos[k,2], pos[k,3]])
        
    return pos


# Function to use a haars cascade and find features
# TODO: We'll want to store our coordinates for output at some later point...
def find_feature(cascade, box_clr, settings, gray, frame, angle):
    feature = cascade.detectMultiScale(gray, **settings)
    
    # Draw a rectangle around the features
    if len(feature) > 0:
        # Flag that we found something
        feature_found = True 
        
        # Rotate our boxes based off of the original frame dimensions if-needed
        feature = rotate_point(feature, frame, -angle)
        
        # Draw the box
        for (x, y, w, h) in feature:
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_clr, box_line_width)
    else:
        # Flag that we didn't find anything
        feature_found = False
        
    return frame, feature_found, feature


# Other functions as-needed
# Function to check if our coordinated are within bounds or if we need to move
# def within_bounds(coord_list, bounds)
    # We could either look at the moving average with a window that's of length N:
    #   https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-numpy-scipy
    # or perhaps we want to just check if the mean/median of the list coordinates
    # is within bounds; we'll probably need to do some experimentation to see what works best
    
    # loop through each feature
        # check to see that the feature is within our bounding box
        # return True, 0 if we're within bounds, return false, amount to move if oob
    # return True
    
# Function to move our motors if we're out of bounds
#def motor_movement(parameters):
    #calculate distance needed to move by motor
    #convert to radians/measure
    #return values to be read by servos
    

############################ Do the main loop ############################

# Open a connection to the webcam    
video_capture = cv2.VideoCapture(0)

# Main loop; this runs forever now, but we can tie it to a hardware switch that
# will make it stop
while True: 
    # Pull a frame from the camera && convert to grayscale
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # IMPORTANT! We ONLY want to call find_feature if we haven't found anything
    # so that we minimize the amount of processing.  We can determine later if
    # we still want to process multiple cascades for better error checking/
    # redundancy down the road.
    
    # Look for faces
    angle = 0
    found = False
    frame, found, coords = find_feature(face_cascade, face_clr, face_settings, gray, frame, angle)
    if not(found):
        # If we didn't find anything without rotating our frames, let's 
        # check for face profiles
        frame, found, coords = find_feature(prof_cascade, prof_clr, prof_settings, gray, frame, angle)
        
        if not(found):
            # If we STILL didn't find anything, let's continue to search
            # first for regular faces, then face profiles at each angle.  
            # We begin by rotating the image, then checking each cascade.
            for angle in angs2scan:
                # print(angle) # for debugging to make sure that we break the
                # loop as soon as we find a feature
                gray_rot = rotate_image(gray, angle)
                
                # Check rotated image for faces
                frame, found, coords = find_feature(prof_cascade, prof_clr, prof_settings, gray_rot, frame, angle)
                if found:
                    # If we found something, we can break the FOR loop so
                    # that we don't need to perform the other rotations/scans
                    break
                else:
                    # Check the rotated image for profiles
                    frame, found, coords = find_feature(prof_cascade, prof_clr, prof_settings, gray_rot, frame, angle)
                    if found: break
            
            
    # If we still haven't found anything, let's check for eyes
    if not(found):
        angle = 0
        frame, found, coords = find_feature(eyes_cascade, eyes_clr, eyes_settings, gray, frame, angle)
    
    # TODO: Finally, check if we're within bounds and move if needed...
    # 1. Collapse our list into the typical value (probably the mean or median of the list),
    #       add it into a queue of coordinates
    # 2. Add the collapsed coordinates to a queue or something like that
    # 3. Check if we're within bounds
    # 4. If within bounds, add a green boundary box, otherwise add a red 
    #       boundary box and reset our queue.  We'll eventually replace 
    #       these boxes with our motor_movement() and/or other function(s)
    
    
    # Display our detected features (this is only for our debugging at this point)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Closing
video_capture.release()
cv2.destroyAllWindows()
