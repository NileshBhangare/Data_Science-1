# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:44:55 2024

@author: Nilesh
"""

import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()
dir(iris)

df=pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()
df['target']=iris.target

df.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df.drop(['target'],axis='columns'),iris.target,test_size=0.2)
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
