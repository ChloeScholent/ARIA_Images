# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 15:09:55 2025

@author: rouxm
"""
import pickle
import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#### SVM #####

param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
svc=svm.SVC(probability=True)
model = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1, verbose=2)
print(model)

Categories=['Squat','Bench','Deadlift']

flat_data_arr=[]
target_arr=[]

datadir=r"C://Users//rouxm//Desktop//2025-2026 - ARIA//Image//Porjet//SVM//pasunDonnées//"

for i in Categories: 
    print(f'loading... category : {i}')
    path=os.path.join(datadir,i)
    for img in os.listdir(path):
        img_array=imread(os.path.join(path,img))
        img_resized = resize(img_array, (64, 64))  
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)
print(target)
df=pd.DataFrame(flat_data)
print(len(df))
df['Target']=target
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)


print('Splitted Successfully')

print('Starting the training...')
model.fit(x_train,y_train)
print('The Model is trained well with the given images')
model_name= "SVM_images.pickle"
#pickle.dump(model, open(filename,"wb"))

# model.best_params_ contains the best parameters obtained 

y_pred=model.predict(x_test)
cm= confusion_matrix(y_test,y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.savefig("TEST.pdf")
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
