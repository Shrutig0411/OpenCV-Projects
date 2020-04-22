# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:01:15 2020

@author: Shruti
"""
#Car_Detection

import time
import cv2
import numpy as np

#Create our car classifier
car_classifier=cv2.CascadeClassifier(r"C:\Users\Shruti\Desktop\Haarcascades\haarcascade_car.xml")

#Initiate video capture
cap=cv2.VideoCapture("C:/Users/Shruti/Desktop/images/cars.avi")

#Loop once video is successfully loaded
while True:
    #time.sleep(.05)
    #Read first frame
    ret, frame=cap.read()
    
    grey=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Pass frame to body classifier
    cars=car_classifier.detectMultiScale(grey, 1.2, 3)
    
    #Extract bounding boxes from any bodies identified
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,127, 255), 2)
        cv2.imshow("Cars", frame)
    
    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()