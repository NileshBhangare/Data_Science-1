# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:06:43 2024

@author: Nilesh
"""
#Assignment On Company Data Data

import pandas as pd

#importing dataset

data=pd.read_csv('c:/2-datasets/Company_Data.csv')
data.head()
data=data.drop('US',axis='columns')

#Separating target and features i.e inputs

x=data.drop('Urban',axis='columns')
y=data['Urban']

#applying onehotencoding to data points
from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder()
#y_encoded=encoder.fit_transform
x_encoded=pd.get_dummies(x)

#Spliting dataset into training and testing

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_encoded,y,test_size=0.2)

#applying random forest model

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()

model.fit(x_train,y_train)
#Checking Acuraccy

model.score(x_test,y_test)
y_predicted=model.predict(x_test)

#generating confusion Matrix

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)
cm

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel("Truth")
