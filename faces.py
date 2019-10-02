import cv2
import numpy as np
from PIL import Image
import os
import pickle


face_cascade = cv2.CascadeClassifier('/Users/rohanpatil/PycharmProjects/opencv/venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
recogniser = cv2.face.LBPHFaceRecognizer_create()
recogniser.read('/Users/rohanpatil/PycharmProjects/opencv/trainer101.yml')


labels = {}


with open('/Users/rohanpatil/PycharmProjects/opencv/labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}
    print(labels)




cap = cv2.VideoCapture(0)

while(True):

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #print(x,y,h,w)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recogniser.predict(roi_gray)
        if conf >= 45 and conf<=85:
            print(id_, conf)
            print(labels[id_])
            name = labels[id_]
            font = cv2.FONT_HERSHEY_COMPLEX
            color = (255,255, 2)
            stroke = 1
            cv2.putText(frame, name, (x,y), font, 1, color,stroke, cv2.LINE_AA)
            #print('0')
        #img_item = '/Users/rohanpatil/PycharmProjects/opencv/venv/face-recognition-opencv/dataset/Rohan Patil/my-img.jpg'
        #cv2.imwrite(img_item, roi_gray)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()