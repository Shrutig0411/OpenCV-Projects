# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 19:39:23 2020

@author: Shruti
"""
#Face and eye detection

import cv2
import numpy as np

#We point Open CV's cascade classifier function to where
#our classifier(xml format) is stored
face_classifier=cv2.CascadeClassifier(r"C:\Users\Shruti\Desktop\Haarcascades\haarcascade_frontalface_default.xml")
eye_classifier=cv2.CascadeClassifier(r"C:\Users\Shruti\Desktop\Haarcascades\haarcascade_eye.xml")

#Load the image and convert it to grey scale
img=cv2.imread("C:/Users/Shruti/Desktop/images/J.jpg")
grey=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Classifier returns the ROI of detected face as a tuple 
#which stores the top left and bottom right coordinates
faces=face_classifier.detectMultiScale(grey, 1.3, 5)

#When no face detected face classifier returns an empty tuple
if faces is ():
    print("No faces found")

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,127, 255), 2)
    cv2.imshow("Face Detection", img)
    cv2.waitKey()
    #Cropping the face part from the image 
    roi_grey=grey[y:y+h, x:x+w]
    roi_color=img[y:y+h, x:x+w]
    eyes=eye_classifier.detectMultiScale(roi_grey)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (127,0, 255), 2)
        cv2.imshow("Face Detection", img)
        cv2.waitKey()

cv2.destroyAllWindows()