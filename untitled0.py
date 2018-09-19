# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 13:39:57 2018

@author: MOHAMMAD AMIN
"""

import cv2

import numpy as np

from matplotlib import pyplot as plt



image = cv2.imread('D:/University/Semester8/CV lab/week5/aaa.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite('D:/University/Semester8/CV lab/week5/gray.jpg', gray_image)

kernel = cv2.getGaussianKernel(9, 1)

twoD = cv2.sepFilter2D(gray_image, -1, kernel, kernel)

cv2.imshow("Gaussian Blur", twoD)

cv2.waitKey(0)

laplacian = cv2.Laplacian(gray_image,cv2.CV_64F)


sobelx = cv2.Sobel(gray_image,-1,1,0)


sobely = cv2.Sobel(gray_image,-1,0,1)

plt.subplot(2,2,1),plt.imshow(gray_image,cmap = 'gray')

plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')

plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')

plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')

plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()

