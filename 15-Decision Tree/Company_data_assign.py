

#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Thu Feb  1 15:06:16 2024

@author:  Nilesh

'''1.	A cloth manufacturing company is interested to know about the different attributes contributing 
to high sales. Build a decision tree & random forest model with Sales as target variable 
(first convert it into categorical variable).

Buisiness understanding:
The business objective seems to involve analyzing company data to predict whether
 a company is based in the US or not.
 
Maximize:
         1. Accuracy   
Minimize:
        1.False positive
        2. false Negative

'''
"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df=pd.read_csv('c:/10-ML/DecisionTree/Company_Data.csv')
df
df.head(10)

df.describe()
'''
Sales   CompPrice      Income  ...       Price         Age   Education
count  400.000000  400.000000  400.000000  ...  400.000000  400.000000  400.000000
mean     7.496325  124.975000   68.657500  ...  115.795000   53.322500   13.900000
std      2.824115   15.334512   27.986037  ...   23.676664   16.200297    2.620528
min      0.000000   77.000000   21.000000  ...   24.000000   25.000000   10.000000
25%      5.390000  115.000000   42.750000  ...  100.000000   39.750000   12.000000
50%      7.490000  125.000000   69.000000  ...  117.000000   54.500000   14.000000
75%      9.320000  135.000000   91.000000  ...  131.000000   66.000000   16.000000
max     16.270000  175.000000  120.000000  ...  191.000000   80.000000   18.000000

[8 rows x 8 columns]
'''

df.info

a=df.isnull()
a.sum()
'''
Sales          0
CompPrice      0
Income         0
Advertising    0
Population     0
Price          0
ShelveLoc      0
Age            0
Education      0
Urban          0
US             0
'''

b=df.isna()
b.sum()
# the given data set don't have an any na Values


# Label Encoding for categorical variables
from sklearn.preprocessing import LabelEncoder
le_ShelveLoc = LabelEncoder()
le_Urban = LabelEncoder()
le_US = LabelEncoder()

df['ShelveLoc'] = le_ShelveLoc.fit_transform(df['ShelveLoc'])
df['Urban'] = le_Urban.fit_transform(df['Urban'])
df['US'] = le_US.fit_transform(df['US'])

# Separate features and target variable
from sklearn.tree import DecisionTreeRegressor
X = df.drop("Sales", axis='columns')
y = df["Sales"]

# Decision Tree model
model = DecisionTreeRegressor()
model.fit(X, y)

# Example predictions
# Assuming ShelveLoc=2, Urban=1, US=0 for the first example
prediction_1 = model.predict([[115, 95, 5, 110, 117, 26, 20, 0, 1, 1]])

# Assuming ShelveLoc=2, Urban=1, US=1 for the second example
prediction_2 = model.predict([[165, 70, 8, 200, 130, 33, 20, 1, 0, 1]])

print("Prediction 1:", prediction_1)
print("Prediction 2:", prediction_2)