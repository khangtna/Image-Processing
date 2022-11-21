import cv2
import numpy as np
import sqlite3
import os



# load tv
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # captureDevice = camera


sampleNum = 0

while (True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if not os.path.exists('dataSet'):
            os.makedirs('dataSet')

        sampleNum += 1


        cv2.imwrite('dataSet/khang/img.' + str(sampleNum) + ' .jpg', gray[y: y + h, x: x + w])
        # cv2.imwrite('a.'+ str(sampleNum) + ' .jpg', gray[y: y + h, x: x + w])

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

    if sampleNum > 200:
        break

cap.release()
cv2.destroyAllWindows()





    








