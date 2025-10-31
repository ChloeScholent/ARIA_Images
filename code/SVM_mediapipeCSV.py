# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 15:09:55 2025

@author: rouxm
"""

import pandas as pd
import mediapipe as mp
import os
import cv2
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#### SVM #####

param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
svc=svm.SVC(probability=True)
model = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1, verbose=2)
print(model)

Categories=['squat','bench','deadlift']

flat_data_arr=[]
target_arr=[]

datadir=r"C:\Users\rouxm\Desktop\2025-2026 - ARIA\Image\SqueezeNet\train"
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

for i in Categories: 
    print(f'loading... category : {i}')
    path=os.path.join(datadir,i)
    for img in os.listdir(path):
        image_path=os.path.join(path,img)
        img_path=path+"\\"+img
        print("reading",image_path)
    
        img=cv2.imread(image_path)
        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        Liste=[[],[],[]] #x, y, z, len(Liste[i])==33
        with mp_pose.Pose(static_image_mode=True,
                               min_detection_confidence=0.7,
                               min_tracking_confidence=0.7) as pose:
            result=pose.process(img_rgb)
            if result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    img_rgb,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                    )
                for lm in result.pose_landmarks.landmark:
                    Liste[0].append(lm.x)
                    Liste[1].append(lm.y)
                    Liste[2].append(lm.z)
                L=[]
                for truc in range (33):
                    L.extend([Liste[0][truc],Liste[1][truc],Liste[2][truc]])
                    print(np.shape(L))
        flat_data_arr.append(L)
        target_arr.append(Categories.index(i))
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)

df=pd.DataFrame(flat_data)
print(df)
df['Target']=target
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)


print('Splitted Successfully')

print('Starting the training...')
model.fit(x_train,y_train)
print('The Model is trained well with the given images')
model_name= "SVM_Mediapiped_images.pickle"
#pickle.dump(model, open(filename,"wb"))
# model.best_params_ contains the best parameters obtained 
y_pred=model.predict(x_test)
cm= confusion_matrix(y_test,y_pred)
ConfusionMatrixDisplay(cm).plot()
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")