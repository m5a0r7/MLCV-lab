# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 14:06:05 2018

@author: MOHAMMAD AMIN
"""


import scipy.io

import numpy as np


import warnings
warnings.filterwarnings("ignore")


from numpy import linalg as LA

import pylab as plt

from sklearn import svm


import cv2

import sklearn



mat1 = scipy.io.loadmat('D:/University/Semester8/CV lab/week2/Yale.mat')

mat1 = mat1["people_illum_img"]




people = 10

train = 8

test = 1 


test_sample = mat1[: , 8 , : , :]

train_sample = mat1[: , 0:train , : , :]


mat2 = mat1.reshape(10 , 9 , 192 * 168)


mat3 = mat2.reshape(90 , 192*168)


pr = [0] * 10

x = 0

y = 0

for i in range(9):
    
    #tr = np.delete(mat2 , i , 1)
    tr = np.concatenate((mat2[:,:i,:], mat2[:,i+1:,:]), 1)
    te = mat2[: , i , :]
    
    tr = tr.reshape(80 , 192*168)
    
    te = te.reshape(10 , 192*168)
    
    tr_lable = [0] * 8 + [1] * 8 +[2] * 8 +[3] * 8 +[4] * 8 +[5] * 8 +[6] * 8 +[7]*8 + [8] * 8 +[9] * 8 
    
    te_lable = [0,1,2,3,4,5,6,7,8,9]
    
    clf = svm.SVC(kernel = "linear")
    
    clf.fit(tr, tr_lable)
    
    pr = clf.predict(te)
    
    w = sklearn.metrics.confusion_matrix([0,1,2,3,4,5,6,7,8,9] , pr)
    
    x = x + np.trace(w)
    
    print(clf.score(te , [0,1,2,3,4,5,6,7,8,9]))
    
    clf.score(te , [0,1,2,3,4,5,6,7,8,9])
    
    
    y = y + clf.score(te , [0,1,2,3,4,5,6,7,8,9])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    