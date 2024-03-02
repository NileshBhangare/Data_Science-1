# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:57:04 2024

@author: Nilesh
"""

'''
Business Understanging:
    Maximize: Accuracy of prediction
    Minimize: Error while Predicting
              Model complexity and overfitting of model
    
    Business Constraints: data security and privacy as persons personal data \
        is given to model
        
        
Data Dictionary

Name Of feature                  Description             Type           Relevance
Number of times pregnant         No of times pregnant    Discreat
Plasma glucose concentration     Glucose Concentration   Discreat
Diastolic blood pressure          
Triceps skin fold thickness
2-Hour serum insulin
Body mass index
Diabetes pedigree function
Age (years)
Class variable
'''
#Assignment On Diabetees Data
import pandas as pd
#importing dataset
diabetes=pd.read_csv('c:/2-datasets/Diabetes.csv')
diabetes

diabetes.head()
diabetes.columns
diabetes.dtypes
#Separating target and features i.e inputs
x=diabetes.drop(' Class variable',axis='columns')
y=diabetes[' Class variable']

#Spliting dataset into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

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
plt.ylabel('Truth')
