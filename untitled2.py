# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:53:29 2018

@author: MOHAMMAD AMIN
"""


import csv

import numpy as np

oz1 = []

oz2 = []

x = []

y = []
n = 107
with open('D:/University/Semester8/CV lab/week1/ozon.csv', 'r') as f:
    
    
    oz = csv.reader(f, delimiter = ',')
    
    for row in oz:
        oz1.append(row)

oz1 = oz1[1:]

oz1 = np.array(oz1 , dtype = 'float')


y = oz1[: , 0]



x = oz1[: , 1:]

x0 = np.ones((111,1))

x = np.concatenate((x0,x), axis = 1)

test_y = y[0:n]

train_y = y[n:]

test_x = x[0:n , :]

train_x = x[n : , :]

B = np.matmul(np.linalg.inv(np.matmul(train_x.T, train_x)), np.matmul(train_x.T, train_y))
#B1 = (np.transpose(train_x) * (train_x))**(-1) * np.transpose(train_x) * train_y





y_t_train = np.matmul(train_x , B)
y_t_test = np.matmul(test_x , B)


err_y_train = y_t_train - train_y

err_y_test = y_t_test - test_y



print(np.sum(np.square(err_y_train))/(111-n))

print(np.sum(np.square(err_y_test))/n)


No = np.random.normal(0 , 1 , (111 , 2))

x_n = x

x_n[:,1:3] = x_n[: , 1:3] + No



test_x_n = x_n[0:n , :]

train_x_n = x_n[n : , :]

B = np.matmul(np.linalg.inv(np.matmul(train_x_n.T, train_x_n)), np.matmul(train_x_n.T, train_y))


y_t_train = np.matmul(train_x_n , B)
y_t_test = np.matmul(test_x_n , B)


err_y_train = y_t_train - train_y

err_y_test = y_t_test - test_y





print(np.sum(np.square(err_y_train))/(111-n))

print(np.sum(np.square(err_y_test))/n)







x_p = x

x_p2 =np.power(x_p[: , 1:] , 2)

x_p = x = np.concatenate((x , x_p2), axis = 1)

test_x_p = x_p[0:n , :]

train_x_p = x_p[n : , :]

B = np.matmul(np.linalg.inv(np.matmul(train_x_p.T, train_x_p)), np.matmul(train_x_p.T, train_y))


y_t_train = np.matmul(train_x_p , B)
y_t_test = np.matmul(test_x_p , B)


err_y_train = y_t_train - train_y

err_y_test = y_t_test - test_y




print(np.sum(np.square(err_y_train))/(111-n))

print(np.sum(np.square(err_y_test))/n)


lam = 0.00001

i = np.identity(4)

B = np.matmul(np.linalg.inv(np.matmul(train_x.T, train_x) + i), np.matmul(train_x.T, train_y))


y_t_train = np.matmul(train_x , B)
y_t_test = np.matmul(test_x , B)


err_y_train = y_t_train - train_y

err_y_test = y_t_test - test_y


print(np.sum(np.square(err_y_train))/(111-n))

print(np.sum(np.square(err_y_test))/n)




