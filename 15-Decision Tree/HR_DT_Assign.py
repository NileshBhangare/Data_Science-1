# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:30:00 2024

@author:  Nilesh
"""

'''4.	In the recruitment domain, HR faces the challenge of predicting if 
the candidate is faking their salary or not. For example, a candidate claims 
to have 5 years of experience and earns 70,000 per month working as a regional manager. 
The candidate expects more money than his previous CTC. We need a way to verify their claims 
(is 70,000 a month working as a regional manager with an experience of 5 years a genuine claim or 
 does he/she make less than that?) Build a Decision Tree and Random Forest model with monthly income 
as the target variable. 

Problem Understanding:
The problem entails using candidate attributes such as years of experience, 
job role (e.g., regional manager), and potentially other factors to predict 
the candidate's monthly income. By analyzing these attributes, HR professionals 
can determine whether a candidate's salary claim aligns with industry standards and 
their claimed level of experience and job role.

Maximize:
    1. Accuracy of salary prediction
    
Minimize:
    1. FP and FN
    2. decisions or negotiations

'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
data=pd.read_csv('c:/10-ML/RandomForest/HR_DT.csv')
data
data.columns

# Separate features (X) and target variable (y)
X = data.drop(' monthly income of employee', axis=1)  # Features
y = data[' monthly income of employee']  # Target variable
X
y
# One-hot encode the target variable
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
#y_encoded = encoder.fit_transform(y.values.reshape(-1, 1)).toarray()
# One-hot encode the features
X_encoded = pd.get_dummies(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_encoded,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=20)

model.fit(X_train,y_train)
model.score(X_test,y_test)
y_predicted=model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')







