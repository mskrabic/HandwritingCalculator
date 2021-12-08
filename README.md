# HandwritingCalculator
An (unsucessful) attempt to build an app for detecting and calculating handwritten math expressions.
Unfortunately, since I haven't managed to train an appropriate model, running the file just prints out the model's prediction for each detected character.
# Instructions
To run use command: <code>python detector.py</code><br/>
It accepts two arguments:<br/>
<code>--visualize=True</code> opens the given image in a new window and draws the bounding boxes on it.</br>
<code>--img=path-to-input-image</code> to set input image. Default is example/test.jpeg

# Components
## solver.py
Simple infix expression solver. Converts the given expression to postfix then evaluates it.

## classifier.py
Loads the trained model stored in the 'bestmodel.h5' file, then uses it for predictions of given inputs.

## detector.py
Detects characters on the given image, passes them to the classifier and calls the solver with its predictions.

## photomath.ipynb
Jupyter notebook describing the training process.
