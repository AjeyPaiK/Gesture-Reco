import cv2
import numpy as np
from model import *
from keras.optimizers import rmsprop
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
cap = cv2.VideoCapture(0)
count = 0
model = model(img_shape=(50, 50, 1))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=rmsprop(lr=0.0001, decay=1e-6), metrics=['accuracy'])
model.load_weights('/home/ajey/Desktop/weights/weights-04-1.00.hdf5')

def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

while(1):
    count += 1
    frameno = str(count)
    #ret returns True if camera is running, frame grabs each frame of the video feed
    ret, frame1 = cap.read()

    frame = cv2.resize(frame1, (50, 50))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame/255.0
    frame = frame.reshape(-1, 50, 50, 1)
    prediction = model.predict(frame, batch_size = None, steps = 1, verbose = 0)
    if np.argmax(prediction) == 0:
        print("Next", prediction[0][0]*100)
        __draw_label(frame1, 'Next', (20,20), (255,0,0))
        __draw_label(frame1, str(int(prediction[0][0]*100))+"%", (20,40), (0,255,0))
    if np.argmax(prediction) == 1:
        print("Previous", prediction[0][1]*100)
        __draw_label(frame1, 'Previous', (20,20), (255,0,0))
        __draw_label(frame1, str(int(prediction[0][1]*100))+"%", (20,40), (0,255,0))
    if np.argmax(prediction) == 2:
        print("Stop", prediction[0][2]*100)
        __draw_label(frame1, 'Stop', (20,20), (255,0,0))
        __draw_label(frame1, str(int(prediction[0][2]*100))+"%", (20,40), (0,255,0))
    out.write(frame1)
    cv2.imshow("Window", frame1)
    if cv2.waitKey(33) & 0xff == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
    elif count == 1000:
        cv2.destroyAllWindows()
        cap.release()
        break
