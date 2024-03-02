# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:22:40 2024

@author: Nilesh
"""

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')
#read csv file
load_data=pd.read_csv('c:/2-datasets/income.csv')
load_data.columns
load_data.head()
#let us split the data in input and output
x=load_data.iloc[:,0:6]
y=load_data.iloc[:,6]
#split the dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#create adaboost classifier
ada_model=AdaBoostClassifier(n_estimators=100,learning_rate=1)
#n_estimators=number of weak learner
#Learning rate,it contributes weights of weak learners,by deafault it is 1
#train the model
model=ada_model.fit(x_train,y_train)
#predict the results
y_pred=model.predict(x_test)
print('accuracy',metrics.accuracy_score(y_test,y_pred))
#let us try for another base model
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
#here base model is trained
Ada_model=AdaBoostClassifier(n_estimators=50,base_estimator=lr,learning_rate=1)
model=Ada_model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print("Accuracy",metrics.accuracy_score(y_test,y_pred))
