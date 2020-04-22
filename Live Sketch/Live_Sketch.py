# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:00:17 2020

@author: Shruti
"""
#Live sketch using webcam

import numpy as np
import cv2

#Our sketch generating function
def sketch(image):
    #Convert image to gray scale
    img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Cleaning up the image, reomve noise using gaussian blur
    blur=cv2.GaussianBlur(img, (5,5), 0)
    #Extract edges
    canny=cv2.Canny(blur, 20,70)
    #Do an invert binarize the image
    ret,mask=cv2.threshold(canny, 70, 255, cv2.THRESH_BINARY_INV)
    return mask

#cap is the object provided by video capture

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    cv2.imshow("Live Sketch", sketch(frame))
    if cv2.waitKey(1)==13:
        break

#Release camera and close windows
cap.release()
cv2.destroyAllWindows()