import cv2
from model import DetectingMask
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = DetectingMask("model_json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()

        gray_fr = cv2.cvtColor(fr, cv2.IMREAD_GRAYSCALE)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)
        output_img = cv2.cvtColor(gray_fr, cv2.COLOR_RGB2BGR)

        for (x, y, w, h) in faces:
            fc = output_img[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_mask(roi[np.newaxis, :, :])

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
