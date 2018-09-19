# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:04:41 2018

@author: MOHAMMAD AMIN
"""


num = int(input("Please enter a number for reconstruct picture"))

import scipy.io

import numpy as np

from numpy import linalg as LA

import pylab as plt

import sklearn.preprocessing 

mat = scipy.io.loadmat('D:/University/Semester8/CV lab/week2/Yale.mat')




mat = mat["people_illum_img"]
mat = mat.reshape(90, 192, 168)

mat = mat.reshape(90 , 192*168)

mat = np.transpose(mat)


k = mat.mean(1)

k = k.reshape(32256 ,1)

k1 = np.repeat(k,90,axis=1)



mat_n = mat - k1

kernel = np.matmul(np.transpose(mat_n) , mat_n)

w , v = LA.eig(kernel)



sorted_indexes = w.argsort()
sorted_indexes = sorted_indexes[::-1]
w = w[sorted_indexes]


v = mat_n @ v[:, sorted_indexes]








y = sum(w)

f = 0

for i in range(90):
    t = w[0:i]
    
    x = sum(t)
    
    if (x/y) > 0.99:
        f = i
        break
    
    
    
    

n = 15


v = v[:, :n]

norms = np.linalg.norm(v, axis=0)

for i in range(n):
    
    v[:, i] = v[:, i] / norms[i]

face = np.matmul(np.transpose(v) , mat_n)
    
#face = sklearn.preprocessing.normalize(face, norm='l2', axis=1, copy=True, return_norm=False)
        
  
    
w_r = face[:, num].transpose() @ v.transpose()    
    



plt.figure()

im = plt.imshow(w_r.reshape(192 , 168) , cmap = 'gray')

plt.figure()

im = plt.imshow(mat[: , num].reshape(192 , 168) , cmap = 'gray')


