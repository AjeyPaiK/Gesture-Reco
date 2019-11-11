import cv2
import numpy as np
#starts the webcam, uses it as video source
cap = cv2.VideoCapture(0) #uses webcam for video
count = 0
while(1):
    count += 1
    frameno = str(count)
    #ret returns True if camera is running, frame grabs each frame of the video feed
    ret, frame = cap.read()
    frame = cv2.resize(frame, (50, 50))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("/home/ajey/Desktop/stop/image"+frameno+".jpg", frame)
    cv2.imshow("Window", frame)
    print(count)
    if cv2.waitKey(33) & 0xff == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
    elif count == 4000:
        cv2.destroyAllWindows()
        cap.release()
        break
