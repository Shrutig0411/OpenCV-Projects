# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:38:28 2020
i
@author: Shruti
"""

import numpy as np
import cv2

#Load input image and convert it to grey scale
img=cv2.imread("C:/Users/Shruti/Desktop/images/Crowd.jpg")
cv2.imshow("Where's the girl", img)
cv2.waitKey()

grey=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Load template image
template=cv2.imread("C:/Users/Shruti/Desktop/images/Crowd2.jpg", 0)

 
res=cv2.matchTemplate(grey, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc=cv2.minMaxLoc(res)

#Create bounding box
top_left=max_loc
bottom_right=(top_left[0]+30, top_left[1]+80)
cv2.rectangle(img, top_left, bottom_right, (0,255,0),2)

cv2.imshow("Here she is", img)
cv2.waitKey()
cv2.destroyAllWindows()
           
