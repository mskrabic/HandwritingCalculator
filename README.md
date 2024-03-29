# HandwritingCalculator
An attempt to build an app for detecting and calculating handwritten math expressions.
Unfortunately, I haven't been able to train a model with which I would be completely satisfied, but nonetheless it was a fun learning experience.
Final implementation has trouble detecting '-', '(' and ')', but works quite well with expressions that do not use these symbols. Some cherry-picked examples are available in the examples folder :)
## Instructions
Two ways to run:
### 1) CMD
use command: <code>python calculator.py</code><br/>
It accepts two arguments:<br/>
<code>--visualize=True</code> opens the given image in a new window and draws the bounding boxes on it.</br>
<code>--img=path-to-input-image</code> to set input image. Default is examples/test.jpeg

### 2) Flask
Run <code>python app.py</code> from the flask-app directory <br/>
Simple web-page for uploading pictures of math expressions which are then fed to the CNN model.
It runs on <code>localhost:5000</code>

## Components
### solver.py
Simple infix expression solver. Converts the given expression to postfix then evaluates it.

### classifier.py
Loads the trained model stored in the 'bestmodel.h5' file, then uses it for predictions of given inputs.

### detector.py
Detects characters on the given image and returns their cropped bounding boxes.

### photomath.ipynb
Jupyter notebook describing the training process.
