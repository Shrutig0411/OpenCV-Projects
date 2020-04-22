# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:32:25 2020

@author: Shruti
"""
#Pedestrian Detection

import cv2
import numpy as np

#Create our body classifier
body_classifier=cv2.CascadeClassifier(r"C:\Users\Shruti\Desktop\Haarcascades\haarcascade_fullbody.xml")

#Initiate video capture
cap=cv2.VideoCapture("C:/Users/Shruti/Desktop/images/walking.avi")

#Loop once vide is successfully loaded
while True:
    #Read first frame
    ret, frame=cap.read()
    #Resizing the frame to increase the speed of classification method 
    #Because larger windows have more pixels to slide over hence will take more time
    frame=cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    
    grey=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Pass frame to body classifier
    bodies=body_classifier.detectMultiScale(grey, 1.2, 3)
    
    #Extract bounding boxes from any bodies identified
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,127, 255), 2)
        cv2.imshow("Pedestrians", frame)
    
    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()

