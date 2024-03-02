# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:33:02 2024

@author: Nilesh
"""

import pandas as pd
hr=pd.read_csv('c:/2-datasets/HR_DT.csv')
hr.head()
#separating target and inputs
x=hr.drop(' monthly income of employee',axis='columns')
y=hr[' monthly income of employee']

#applying onehot encoding
from sklearn.preprocessing import OneHotEncoder
x_encoded=pd.get_dummies(x)

#applying random forest model

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_encoded,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)
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
plt.ylabel('Truth')
