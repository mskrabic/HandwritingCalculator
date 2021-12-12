import detector
import classifier
import solver
import argparse
import cv2 as cv


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default='examples/test.jpeg', type=str)
    parser.add_argument('--visualize', default=False, type=bool)
    args = parser.parse_args()
    calculator = Calculator(args.visualize)
    img = cv.imread(args.img)
    calculator.calculate(img)

class Calculator():
    def __init__(self, visualize=False):
        self.solver = solver.Solver()
        self.detector = detector.CharacterDetector(visualize)
        self.classifier = classifier.HandwritingClassifier()

    def calculate(self, img):
        crops = self.detector.detect(img)
        expression = ''
        for crop in crops:
            expression = expression + self.classifier.run(crop) + ' '
        result = self.solver.evaluate(expression)
        if result is not None:
            print(f'{expression} = {result}')



if __name__ == '__main__':
    run()