import cv2
import numpy as np

modelFile = "/home/pi/Downloads/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "/home/pi/Downloads/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
skipframe = True

while True:
    ret, img = video_capture.read()
    
    if skipframe:
        img2 = img
        blob = cv2.dnn.blobFromImage(img2, 1.0, (175, 175), (104.0, 117.0, 123.0))
        net.setInput(blob)
        faces = net.forward()
        #to draw faces on image
        h, w = img2.shape[:2]
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(img2, (x, y), (x1, y1), (0, 0, 255), 2)

    skipframe = not skipframe
    cv2.imshow('Video', img2)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
video_capture.release()
cv2.destroyAllWindows()
