import dlib
from imutils import face_utils
import cv2
import time

#def detect_face_on_video_dlib(video_source):
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
detector = dlib.get_frontal_face_detector()
#frame_process = True
counter = 1
while True:    
    # Read video stream
    ret, original_frame = cap.read()

        #frame = cv2.resize(frame, (640, 480))

        # Convert into grayscale
        #img = frame.copy()
    
    frame = cv2.resize(original_frame, (0, 0), fx=0.4, fy=0.4)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
        # Detect faces
    t1 = time.perf_counter()
    faces = detector(frame, 1)
    t2 = time.perf_counter()
    print(f'{counter}: {1/(t2-t1):.2f} Hz')
                
        # Loop through list (if empty this will be skipped) and overlay green bboxes
        # Format of bboxes is: xmin, ymin (top left), xmax, ymax (bottom right)
    for i in faces:
        (x, y, w, h) = face_utils.rect_to_bb(i) 
        xmin = x 
        ymin = y
        xmax = x + w
        ymax = y + h
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

    cv2.imshow("faces", frame)
    #frame_process = not frame_process
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
        
    # Release everything if job is finished
cap.release()
cv2.destroyAllWindows()

