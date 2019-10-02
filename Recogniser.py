import cv2
import numpy as np
from PIL import Image
import os
import pickle

face_cascade = cv2.CascadeClassifier('/Users/rohanpatil/PycharmProjects/opencv/venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
recogniser = cv2.face.LBPHFaceRecognizer_create()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'face-recognition-opencv')




x_train = []
y_labels = []
label_ids = {}
current_ids = 0




for (root, dirs, files) in os.walk(os.path.dirname('/Users/rohanpatil/PycharmProjects/opencv/venv/face-recognition-opencv/dataset')):
    for file in files:
        if file.endswith("jpg") or file.endswith("png") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(' ', '-')
            print(label, path)

            if not label in label_ids:
                label_ids[label] = current_ids
                current_ids += 1

            id_ = label_ids[label]
            print(label_ids)
            #x_train.append(path)
            #y_label.append(label)
            pil_image = Image.open(path).convert("L")  # Grayscale
            image_array = np.array(pil_image, "uint8")
            print(image_array)
            faces = face_cascade.detectMultiScale(image_array, 1.5, 5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

print(y_labels, x_train)
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recogniser.train( x_train, np.array(y_labels))
recogniser.save("trainer101.yml")
