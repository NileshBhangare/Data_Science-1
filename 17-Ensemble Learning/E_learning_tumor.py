# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:51:28 2024

@author: Nilesh
"""
''' 
Problem Statement: Most cancers form a lump called a tumour. 
                But not all lumps are cancerous. Doctors extract a sample 
                from the lump and examine it to find out if it’s cancer or 
                not. Lumps that are not cancerous are called benign (be-NINE).
                Lumps that are cancerous are called malignant (muh-LIG-nunt).
                Obtaining incorrect results (false positives and false 
                negatives) especially in a medical condition such as cancer 
                is dangerous. So, perform Bagging, Boosting, Stacking, and 
                Voting algorithms to increase model performance and provide 
                your insights in the documentation.
                
                
                
Business Objective:
    Maximize:
    Minimize:
        
Business Constraints:
    
'''
import pandas as pd

tumor=pd.read_csv('c:/17-Ensemble Learning/Tumor_Ensemble.csv')
tumor.columns
tumor.head()
tumor.shape
#it has 569 rows and 32 columns

#Cheking for duplicates in dataset
tumor.duplicated().sum()
#There is no any duplicate record

#Checking for null values
tumor.isnull().sum()
#There is no any null vales in dataset

#5 Number summary
des=tumor.describe()
#All features are having different scales

import seaborn as sns
sns.boxplot(tumor)

#scaling all features using robustscaler
from sklearn.preprocessing import RobustScaler
target=tumor['diagnosis']
target=pd.get_dummies(target)
target.drop(columns=['M'],axis=1,inplace=True)
tumor.drop(columns=['diagnosis','id'],axis=1,inplace=True)

scaler=RobustScaler()
scaled_data=scaler.fit(tumor)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')
#read csv file
#let us split the data in input and output
x=tumor.iloc[:,0:30]
#split the dataset
x_train,x_test,y_train,y_test=train_test_split(x,target,test_size=0.2)
#create adaboost classifier
ada_model=AdaBoostClassifier(n_estimators=100,learning_rate=1)
#n_estimators=number of weak learner
#Learning rate,it contributes weights of weak learners,by deafault it is 1
#train the model
model=ada_model.fit(x_train,y_train)
#predict the results
y_pred=model.predict(x_test)
print('accuracy',metrics.accuracy_score(y_test,y_pred))
#It gives accuracy of 98%

#let us try for another base model
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
#here base model is trained
Ada_model=AdaBoostClassifier(n_estimators=50,base_estimator=lr,learning_rate=1)
model=Ada_model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print("Accuracy",metrics.accuracy_score(y_test,y_pred))
#It gives accuracy of 92%