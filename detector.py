import numpy as np
import cv2 as cv
import classifier
import solver
import argparse

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default='examples/test.jpeg', type=str)
    parser.add_argument('--visualize', default=False, type=bool)
    args = parser.parse_args()
    detector = CharacterDetector(args.visualize)
    img = cv.imread(args.img)
    detector.calculate(img)

class CharacterDetector():
    def __init__(self, visualize=False):
        self.classifier = classifier.HandwritingClassifier()
        self.solver = solver.Solver()
        self.visualize = visualize

    def sorted_bbox(self, contours):
        '''
        Returns the bounding boxes, sorted by their x-axis position.
        Merges overlapping bounding boxes (useful for detection of division symbol).
        '''
        rects = []
        rectsUsed = []
        for cnt in contours:
            rects.append(cv.boundingRect(cnt))
            rectsUsed.append(False)
        def getXFromRect(item):
            return item[0]

        rects.sort(key=getXFromRect)
        acceptedRects = []
        xThr = 1
        for supIdx, supVal in enumerate(rects):
            if (rectsUsed[supIdx] == False):
                currxMin = supVal[0]
                currxMax = supVal[0] + supVal[2]
                curryMin = supVal[1]
                curryMax = supVal[1] + supVal[3]
                rectsUsed[supIdx] = True

                for subIdx, subVal in enumerate(rects[(supIdx + 1):], start=(supIdx + 1)):
                    candxMin = subVal[0]
                    candxMax = subVal[0] + subVal[2]
                    candyMin = subVal[1]
                    candyMax = subVal[1] + subVal[3]

                    if (candxMin <= currxMax + xThr):
                        currxMax = max(currxMax, candxMax)
                        curryMin = min(curryMin, candyMin)
                        curryMax = max(curryMax, candyMax)
                        rectsUsed[subIdx] = True
                    else:
                        break
                acceptedRects.append([currxMin, curryMin, currxMax - currxMin, curryMax - curryMin])
        return acceptedRects

    def detect(self, img):
        '''
        Detects all characters in the image and passes them to the classifier to process.
        '''
        copy = img.copy()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 120, 255, cv.THRESH_BINARY_INV)
        canny = cv.Canny(thresh, 100, 200)
        contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        bounding_boxes = self.sorted_bbox(contours)
        crops = []
        for bbox in bounding_boxes:
            rect_x = bbox[0]
            rect_y = bbox[1]
            rect_w = bbox[2]
            rect_h = bbox[3]

            rect_area = rect_w * rect_h
            min_area = 150

            if rect_area > min_area:
                # Draw bounding box:
                if self.visualize:
                    color = (0, 255, 0)
                    cv.rectangle(copy, (int(rect_x), int(rect_y)), (int(rect_x + rect_w), int(rect_y + rect_h)), color, 2)
                    cv.imshow("Bounding Boxes", copy)
                    cv.waitKey(1000)

                # Crop bounding box:
                current_crop = gray[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w]
                crops.append(current_crop)
        return crops;

    def calculate(self, img):
        expression = ''
        crops = self.detect(img)
        for crop in crops:
            expression = expression + self.classifier.run(crop) + ' '
        print(expression)
        #print(self.solver.evaluate(expression))

if __name__ == '__main__':
    run()

