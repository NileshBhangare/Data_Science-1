# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 22:41:47 2024

@author: Nilesh
"""
'''3.	Build a Decision Tree & Random Forest model on the 
fraud data. Treat those who have taxable_income <= 30000 as 
Risky and others as Good (discretize the taxable 
            
Problem Understanding:
The problem involves using financial data, specifically taxable income, 
to predict whether an individual is risky (taxable income <= 30000) or 
good (taxable income > 30000). This is a binary classification task where 
the outcome variable is the risk level, derived from taxable income.
                          
Maximize:
    1. Maximize Detection of Risky Individuals: 
Minimize:
    1. FP and FN

'''
#importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing

df = pd.read_csv("c:/10-ML/DecisionTree/Fraud_check.csv")
df.head()
df.tail()

#Creating dummy vairables for ['Undergrad','Marital.Status','Urban'] dropping first dummy variable
df=pd.get_dummies(df,columns=['Undergrad','Marital.Status','Urban'], drop_first=True)

#Creating new cols TaxInc and dividing 'Taxable.Income' cols on the basis of [10002,30000,99620] for Risky and Good
df["TaxInc"] = pd.cut(df["Taxable.Income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])
print(df)                                         

#Lets assume: taxable_income <= 30000 as “Risky=0” 
#and others are “Good=1”
df.tail(10)

# let's plot pair plot to visualise the attributes all at once
import seaborn as sns
sns.pairplot(data=df, hue = 'TaxInc_Good')

def norm_func(i):
    # Check if the input is numeric
    if pd.api.types.is_numeric_dtype(i):
        x = (i - i.min()) / (i.max() - i.min())
        return x
    else:
        # If the input is not numeric, return it as is
        return i

# Assuming df is your DataFrame
# First, convert the columns to numeric if they are not already numeric
df_numeric = df.apply(pd.to_numeric, errors='ignore')

# Then, apply the normalization function
#df_norm = df_numeric.apply(norm_func, axis=0)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df.iloc[:,1:])
df_norm.tail(10)
df_norm.columns
# Now, df_norm should contain the normalized DataFrame

# Declaring features & target
X = df_norm.drop(['TaxInc'], axis=1)
y = df_norm['TaxInc']

##Converting the Taxable income variable to bucketing. 
df_norm["income"]="<=30000"
df_norm.loc[df["Taxable.Income"]>=30000,"income"]="Good"
df_norm.loc[df["Taxable.Income"]<=30000,"income"]="Risky"

##Droping the Taxable income variable
df.drop(["Taxable.Income"],axis=1,inplace=True)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = le.fit_transform(df[column_name])
    else:
        pass
    
##Splitting the data into featuers and labels
features = df.iloc[:,0:5]
labels = df.iloc[:,5]

## Collecting the column names
colnames = list(df.columns)
predictors = colnames[0:5]
target = colnames[5]

##Splitting the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2,stratify = labels)

##Model building
from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_jobs = 3,n_estimators = 15, oob_score = True, criterion = "entropy")
model.fit(x_train,y_train)
RandomForestClassifier(criterion='entropy', n_estimators=15, n_jobs=3,
                       oob_score=True)
model.estimators_
model.classes_
model.n_features_
model.n_classes_

model.n_outputs_

model.oob_score_
###74.7833%

##Predictions on train data
prediction = model.predict(x_train)

##Accuracy
# For accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train,prediction)
accuracy
##98.33%
np.mean(prediction == y_train)
##98.33%

##Confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_train,prediction)

##Prediction on test data
pred_test = model.predict(x_test)
##Accuracy

acc_test =accuracy_score(y_test,pred_test)
acc_test

