# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:28:40 2018

@author: MOHAMMAD AMIN
"""

from skimage import exposure
from skimage import feature
import cv2


image = cv2.imread('D:/University/Semester8/CV lab/week7/lady-gaga.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

aaa = feature.local_binary_pattern(gray_image , 10 , 5 , method = "default")

cv2.imshow("sdasdasd" , aaa)

cv2.waitKey()