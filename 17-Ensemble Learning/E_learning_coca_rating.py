# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 21:09:14 2024

@author:  Nilesh
"""
'''
Problem Statement: A sample of global companies and their ratings are given 
                    for the cocoa bean production along with the location of 
                    the beans being used. Identify the important features in 
                    the analysis and accurately classify the companies based 
                    on their ratings and draw insights from the data. 
                    Build ensemble models such as Bagging, Boosting, Stacking,
                    and Voting on the dataset given.
                    
Business Objective:
    Maximize:
    Minimize:
        
Business Constraints:


'''
import pandas as pd
#read csv file
coca=pd.read_excel('c:/17-Ensemble Learning/Coca_Rating_Ensemble.xlsx')
coca.columns
coca.head()

#checking for rows and columns 
coca.shape
#It has 1795 rows and 9 columns

#Checking for duplicated records in dataset
coca.duplicated().sum()
#it has 0 duplicate values

#Checking for null values
coca.isnull().sum()
#In dataset there is 2 null values 
# one in Bean_TYpe column and One in Origin column
#Droping null values
coca=coca.dropna()

#again checking for null values
coca.isnull().sum()
#Now there is no any null value in dataset

#5 number summary
des=coca.describe()
#all features are at different scale

import seaborn as sns
sns.boxplot(coca)
#There is no any outlier in dataset

#Finding correlation using heatmap
sns.heatmap(coca, annot=True, cmap='coolwarm', fmt=".2f")

coca.drop(columns=['Company','Name','Company_Location','Bean_Type'],axis=1,inplace=True)
coca_encoded=pd.get_dummies(coca)

#As data is not normally distributed and there is no any outlier
#we can use robustscaler
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
coca_scale=scaler.fit(coca_encoded)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')
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
