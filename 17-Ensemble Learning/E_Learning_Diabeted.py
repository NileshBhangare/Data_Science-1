# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 19:29:49 2024

@author: Nilesh
"""
''' 
Problem Statement: Given is the diabetes dataset. Build an ensemble model to 
                correctly classify the outcome variable and improve your 
                model prediction by using GridSearchCV. You must apply 
                Bagging, Boosting, Stacking, and Voting on the dataset. 
                
                
Business Objective:
    Maximize:
    Minimize:
        
Business Constraints:


'''
import pandas as pd

#read csv file
load_data=pd.read_csv('c:/17-Ensemble Learning/Diabeted_Ensemble.csv')
load_data.columns
load_data.head()

#Checking for duplicates records in dataset
load_data.duplicated().sum()
#There is no duplicated value

#Checking for null value
load_data.isnull().sum()
#There is no null value present in dataset

#Printing 5 number summary
des=load_data.describe()
#By using 5 number summary,all the features are of different scale and 
#we need to scale down to same scale 

#checking for outlier
import seaborn as sns
sns.boxplot(load_data)
#By printing boxplot,it have outliers in all the features

#As dataset contain outliers and all features are of different scale 
#so that using scaler to make it in same scale and removing outliers

from sklearn.preprocessing import RobustScaler
#as robust scaler need only continuos data
#converting target variable to numeric using oneHotEncoding

target=load_data[' Class variable']
target=pd.get_dummies(target)
target.drop(columns=['YES'],axis=1,inplace=True)
target
load_data.drop(columns=[' Class variable'],axis=1,inplace=True)

scaler=RobustScaler()
scaled_data=scaler.fit(load_data)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

#let us split the data in input and output
x=load_data.iloc[:,0:8]
y=target
y
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
#It give accuracy of 79%

#let us try for another base model
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
#here base model is trained
Ada_model=AdaBoostClassifier(n_estimators=50,base_estimator=lr,learning_rate=1)
model=Ada_model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print("Accuracy",metrics.accuracy_score(y_test,y_pred))
#it gives accuracy of 82%
