# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:37:38 2018

@author: MOHAMMAD AMIN
"""

import scipy.io

import numpy as np



from numpy import linalg as LA

import pylab as plt


import cv2



mat1 = scipy.io.loadmat('D:/University/Semester8/CV lab/week2/Yale.mat')

mat1 = mat1["people_illum_img"]


mat = np.zeros((10 , 9 , 48 , 42))

for i in range(10):
    for j in range(9):
        mat[i , j , : , : ] = cv2.resize(mat1[i , j , : , :], None , fx=0.25, fy=0.25) 



people = 4

train = 8

test = 1 

mat1 = mat[0:people , : , : , :]

test_sample = mat1[: , 8 , : , :]

train_sample = mat1[: , 0:train , : , :]




test_sample1 = test_sample.reshape(people , test, 48 * 42)

#test_sample1 = test_sample1.reshape(people * test , 48*42)

test_sample1 = np.transpose(test_sample1)


train_sample1 = train_sample.reshape(people , train , 48 * 42)

#train_sample1 = train_sample1.reshape(people * (train - test) , 48*42)

train_sample1 = np.transpose(train_sample1)


train_sample1_n = np.zeros((2016 , train , people))

test_sample1_n = np.zeros((2016 , test , people))


m = np.mean(train_sample1 , 1)
        
m = m[: , np.newaxis , :]

train_sample1_n = train_sample1 - m
  

m1 =   np.mean(test_sample1 , 1)    
  

m1 = m1[: , np.newaxis , :]
        
 
test_sample1_n = test_sample1 - m1       


train_sample1_n = train_sample1_n.reshape(2016 , people * train)

test_sample1_n = test_sample1_n.reshape(2016 , people * test)


Sw = np.matmul(train_sample1_n , train_sample1_n.T)


m2 = np.mean(m , 2)

m3 = np.mean(m1 , 2)

m = m.reshape(2016 , 4)

m1 = m1.reshape(2016 , 4)

m = m - m2

m1 = m1 - m3

s = [train] * 4

s = np.diag(s)


Sb = np.matmul(m , s)

Sb = np.matmul(Sb , m.T)



Sw_inv = LA.pinv(Sw)

fin = np.matmul(Sw_inv , Sb)

w , v = LA.eig(fin)


v = np.real(v)

w = np.real(w)



w1 = w

eigenValues, eigenVectors = LA.eig(fin)

idx = w.argsort()[::-1] 

  
w = w[idx]


v = v[:,idx]


#norms = np.linalg.norm(v, axis=0)
#for i in range(2016):
    #v[:, i] = v[:, i] / norms[i]

vec = v[: , 0:2]


train_sample3 = train_sample1.reshape(2016 , 32)


test_sample3 = test_sample1.reshape(2016 , 4)


a = np.matmul(vec.T , train_sample3)   

b = np.matmul(vec.T , test_sample3)  


c = ['red','green','blue', 'gray', 'purple']

for i in range(people):      
    plt.scatter(b[0,i:(i+1)], b[1,i:(i+1)], color = c[i], label="test", marker = 'v')
  


  
for i in range(people):
    for j in range(people * train):
        
        x = []
        
        if (j % people) == i:
            x.append(j)
            
    
    
    
        plt.scatter(a[0,x], a[1,x], color = c[i], label="train", marker = 'o')

#plt.scatter(a[0 , :] , a[1, :] , color = 'red', label="train", marker = 'o')

plt.show()

