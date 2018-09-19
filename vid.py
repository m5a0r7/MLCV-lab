# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:39:11 2018

@author: MOHAMMAD AMIN
"""





import numpy as np
import cv2





cap = cv2.VideoCapture(0)
a = True

while(True):
    ret, frame = cap.read()
    print(ret)
    if (ret and a):
        
        
        
        
        frame0 = frame
        
        a = False
        
        cv2.imshow("" , frame0)
        
        cv2.waitKey(1)
        
        break



# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        
    
        
        frame2 = frame - frame0
        
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        ret1 , frame2 = cv2.threshold(frame2, 70, 255,cv2.THRESH_BINARY)
        
        
        kernel = np.ones((5,5),np.uint8)
        
        frame2 = cv2.erode(frame2,kernel,iterations = 2)
        
        #frame2 = cv2.dilate(frame2,kernel,iterations = 1)
        
        
        frame3 = cv2.bitwise_and(frame , frame , mask = frame2)

        # write the flipped frame
        

        cv2.imshow('frame',frame3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()