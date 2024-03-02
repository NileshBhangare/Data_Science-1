# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 21:58:09 2024

@author: Nilesh
"""

import pandas as pd
#read csv file
password=pd.read_excel('c:/17-Ensemble Learning/Ensemble_Password_Strength.xlsx')
password.columns
password.head()

#checking no of rows and columns
password.shape 
#it has 1999 rows and 2 columns

#Checking for duplicated value in dataset
password.duplicated().sum()
#there is no any duplicate value

#Checking for null values
password.isnull().sum()
#there is no any null value

#5 number summary
password.describe()
#the data is normally distributed as data between 0 to 1

#checking for outliers
import seaborn as sns
sns.boxplot(password)
#there is no any outlier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

#let us split the data in input and output
x=password.iloc[:,0:1]
y=password.iloc[:,1]

#split the dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#create adaboost classifier
ada_model=AdaBoostClassifier(n_estimators=100,learning_rate=1)

#n_estimators=number of weak learners
#Learning rate,it contributes weights of weak learners,by default it is 1
#train the model
model=ada_model.fit(x_train,y_train)
110#predict the results
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

