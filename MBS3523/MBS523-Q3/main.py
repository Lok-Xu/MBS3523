import cv2
import imutils
import numpy as np
import argparse
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
car_cascade = cv2.CascadeClassifier('resources/cars.xml')
def detect(frame):
    bounding_box_cordinates, weights = HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(4, 4), scale=1.03)
    for x, y, w, h in bounding_box_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 4)
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('output', frame)
    return frame
cap = cv2.VideoCapture('resources/IMG_1868.mp4')
while True:
    check, frame = cap.read()
    frame = detect(frame)
    frame = cv2.resize(frame, (640, 480))

    if cv2.waitKey(100) == 27:
        break
cap.release()
cv2.destroyAllWindows()
