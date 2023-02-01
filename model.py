import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)


class DetectingMask(object):
    EMOTION_LIST = ['with mask', 'without mask']

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, 'r') as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)

    def predict_mask(self, img):
        self.preds = self.loaded_model.predict(img)
        return DetectingMask.EMOTION_LIST[np.argmax(self.preds)]
