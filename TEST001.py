import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'face-recognition-opencv/dataset')

for (root, dirs, files) in os.walk(os.path.dirname('/Users/rohanpatil/PycharmProjects/opencv/face-recognition-opencv')):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            print(path)