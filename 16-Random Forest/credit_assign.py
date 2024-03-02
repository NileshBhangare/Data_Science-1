# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:43:05 2024

@author:  Nilesh
"""

import pandas as pd
credit=pd.read_csv('c:/2-datasets/credit.csv')
credit.head()
credit.drop('phone',axis='columns')

x=credit.drop('default',axis='columns')
y=credit['default']

from sklearn.preprocessing import OneHotEncoder
x_encoded=pd.get_dummies(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_encoded,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)
model.score(x_test,y_test)
y_predicted=model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)
cm

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')