import cv2
import time
from mtcnn.mtcnn import MTCNN

detector = MTCNN()
counter = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True: 
    #Capture frame-by-frame
    __, frame = cap.read()
    frame = cv2.resize(frame, (0,0), fx=0.45, fy=0.45)
    
    #Use MTCNN to detect faces
    t1 = time.perf_counter()
    result = detector.detect_faces(frame)
    t2 = time.perf_counter()
    print(f'{counter}: {1/(t2-t1):.2f} Hz')
    
    if result != []:
        for face in result:
            bbox = face['box']
            cv2.rectangle(frame,
                          (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1] + bbox[3]),
                          (0,155,255),
                          2)
            print(f'{bbox[2]} x {bbox[3]}')
    
    counter = counter + 1
    #display resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
        
#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()
