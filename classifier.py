import numpy as np
import cv2 as cv
import math
import pickle
import cv2 as cv


class HandwritingClassifier(object):
    def __init__(self, model=1):
        self.MAP_SYMBOLS = {10: '+', 11: '-', 12: '*', 13: '/', 14: '(', 15: ')'}
        if model == 1:
            import keras
            self.model = keras.models.load_model('bestmodel.h5')

    def run(self, img):
        ret, thresh = cv.threshold(img, 130, 255, cv.THRESH_BINARY_INV)
        img = cv.resize(thresh, (28, 28), interpolation=cv.INTER_AREA)

        img = img.astype(np.float32)/255
        img = np.expand_dims(img, -1)
        input = np.array([img])
        result = self.model.predict(input)
        for i in range(len(result)):
            prediction = np.argmax(result[i])
            if int(prediction) > 9:
                prediction = self.MAP_SYMBOLS[int(prediction)]
            else:
                prediction = str(prediction)
        return prediction

