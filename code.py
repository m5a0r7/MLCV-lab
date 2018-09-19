# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 23:56:59 2018

@author: MOHAMMAD AMIN
"""


from sklearn.model_selection import LeaveOneOut
import numpy as np
from scipy.io import loadmat
from sklearn import svm
import sklearn
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
from numpy import linalg as LA
from sklearn.ensemble import RandomForestClassifier


data1 = loadmat('D:/University/Semester8/CV lab/Project/New folder/CAD_60.mat')
data2 = loadmat('D:/University/Semester8/CV lab/Project/New folder/TST.mat')

x = loadmat('D:/University/Semester8/CV lab/Project/New folder/CAD_60.mat')
y = loadmat('D:/University/Semester8/CV lab/Project/New folder/TST.mat')

X = np.array([[1, 2], [3, 4] , [5,6]])
y1 = np.array([1, 2])
loo = LeaveOneOut()

x = data1

skeleton_data = x['skeleton_data']
action_lengths = x['action_lengths']
action_names = x['action_names']
joint_names = x['joint_names']

data = skeleton_data

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        for k in range(data.shape[2]):
            for t in range(data.shape[4]):
                x = skeleton_data[i,j,k,:action_lengths[i,j,k],t,:]
                frame_num = list(np.arange(action_lengths[i,j,k]))
                x1 = x[: , 0]
                x2 = x[: , 1]
                x3 = x[: , 2]
                f1 = interp1d(frame_num, x1 , kind='cubic')
                f2 = interp1d(frame_num, x2, kind='cubic')
                f3 = interp1d(frame_num, x3, kind='cubic')
                frame_num_new = np.linspace(0,action_lengths[i,j,k]-1 , num=data.shape[3], endpoint=True)
                y1 = f1(frame_num_new)
                y2 = f2(frame_num_new)
                y3 = f3(frame_num_new)
                data[i,j,k,:,t,0] = y1
                data[i,j,k,:,t,1] = y2
                data[i,j,k,:,t,2] = y3

for a in range(data.shape[0]):
    for k in range(data.shape[1]):
        for b in range(data.shape[2]):
            for c in range(data.shape[3]):
                for d in range(data.shape[4]):
                    norm = np.sqrt((data[a,k,b,c,2,0] - data[a,k,b,c,1,0])**2 + (data[a,k,b,c,2,1] - data[a,k,b,c,1,1])**2 + (data[a,k,b,c,2,2] - data[a,k,b,c,1,2])**2)
                    for e in range(3):
                        data[a,k,b,c,d,e] = (data[a,k,b,c,d,e] - data[a,k,b,c,2,e])/norm
                  
                  
                    


data = np.nan_to_num(data)


score = 0
score1 = 0
cluster_size = 40
for p in range(data.shape[0]):
    data1 = np.delete(data, p, 0)
    tr = data1
    data1 = data1.reshape((data.shape[0]-1)*data.shape[1]*data.shape[2]*data.shape[3] , data.shape[4]*data.shape[5])
    kmeans = KMeans(n_clusters=cluster_size)
    kmeans.fit(data1)
    hist_of_label = np.zeros(shape = (cluster_size ,1))
    svm_lable = []

    for i in range(data.shape[0]-1):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                svm_lable.append(k)
                action = tr[i,j,k ,:,:,:]
                action = action.reshape(data.shape[3],data.shape[4]*data.shape[5])
                l = kmeans.predict(action)
                l = l.reshape(data.shape[3],1)
                l = list(l)
                count_l = []
                for t in range(cluster_size):
                    count_l.append(l.count(t))
                count_l = np.asarray(count_l)
                count_l = np.transpose(count_l)
                count_l = count_l.reshape(cluster_size,1)
                hist_of_label = np.concatenate((hist_of_label, count_l), axis=1) 
            
    hist_of_label = np.delete(hist_of_label, 0, 1)
    hist_of_label_test = np.zeros(shape = (cluster_size ,1))
    svm_lable_test = []
    for j in range(data.shape[1]):
        for k in range(data.shape[2]):
            action = data[p,j,k ,:,:,:]
            svm_lable_test.append(k)
            action = action.reshape(data.shape[3],data.shape[4]*data.shape[5])
            l = kmeans.predict(action)
            l = l.reshape(data.shape[3],1)
            l = list(l)
            count_l = []
            for t in range(cluster_size):
                count_l.append(l.count(t))
                
                
            count_l = np.asarray(count_l)
            count_l = np.transpose(count_l)
            count_l = count_l.reshape(cluster_size,1)
            hist_of_label_test = np.concatenate((hist_of_label_test, count_l), axis=1) 
            
           
    hist_of_label_test = np.delete(hist_of_label_test, 0, 1)

    hist_of_label_test = np.transpose(hist_of_label_test)



    clf = svm.SVC(kernel = "rbf")
    clf.fit(np.transpose(hist_of_label), np.transpose(svm_lable))
    pr = clf.predict(hist_of_label_test)
    
    w = sklearn.metrics.confusion_matrix(np.transpose(svm_lable_test) , pr)
    
    
    
    print("clf.score is    " , clf.score(hist_of_label_test , np.transpose(svm_lable_test)))
    clf1 = RandomForestClassifier(max_depth=25, random_state=0)
    
    score = score + clf.score(hist_of_label_test , np.transpose(svm_lable_test))
    clf1.fit(np.transpose(hist_of_label), np.transpose(svm_lable))
    print("clf.score for random forest is    " , clf1.score(hist_of_label_test , np.transpose(svm_lable_test)))
    score1 = score1 + clf1.score(hist_of_label_test , np.transpose(svm_lable_test))


print("The accuracy is foe svm is:   "  , score/data.shape[0])
print("The accuracy for random forest is:   "  , score1/data.shape[0])
