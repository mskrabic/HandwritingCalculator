import numpy as np
import cv2 as cv
import math
import pickle
import cv2 as cv
import utils


class HandwritingClassifier(object):
    def __init__(self, model=1):
        self.MAP_SYMBOLS = {10: '+', 11: '-', 12: '*', 13: '/', 14: '(', 15: ')'}
        if model == 1:
            import keras
            self.model = keras.models.load_model('bestmodel.h5')

    def run(self, img):
        img = utils.prep_img(img)
        img = img.astype(np.float32)/255.0
        img = np.expand_dims(img, -1)

        input = np.array([img])
        result = self.model.predict(input)[0]
        prediction = np.argmax(result)
        if prediction > 9:
            prediction = self.MAP_SYMBOLS[prediction]
        else:
            prediction = str(prediction)
        return prediction

