import os
from flask import Flask, render_template, request, url_for
from flask import send_from_directory
import sys

import cv2 as cv
import numpy as np
sys.path.append(os.path.abspath(".."))
import utils
import detector
import classifier
import solver

UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = -1
detector = detector.CharacterDetector()
classifier = classifier.HandwritingClassifier()
solver = solver.Solver()


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)

    image = cv.imread(full_name)
    crops, copy, bboxes = detector.detect(image)
    expression = ''
    for i, crop in enumerate(crops):
        prediction = classifier.run(crop)
        expression = expression + prediction + ' '
        cv.putText(copy, prediction, (bboxes[i][0] + 20, bboxes[i][1]-20), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    filename = file.filename[:file.filename.rindex('.')] + '_new' + file.filename[file.filename.rindex('.'):]
    full_name = os.path.join(UPLOAD_FOLDER, filename)
    cv.imwrite(full_name, copy)
    result = solver.evaluate(expression)

    return render_template('upload.html', image_file_name=filename, expression=expression, result=result)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
