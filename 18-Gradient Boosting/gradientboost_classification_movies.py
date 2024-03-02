# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:07:30 2024

@author:  Nilesh
"""

import pandas as pd

df=pd.read_csv('c:/16-Ensemble Learning/movies_classification.csv')

#dummy variables
df.head()
df.info()

#n-1 dummy variables will be created for n categories
df=pd.get_dummies(df,columns=['3D_available','Genre'],drop_first=True)

df.head()

#Input and output split
predictors=df.loc[:,df.columns!='Start_Tech_Oscar']
type(predictors)

target=df['Start_Tech_Oscar']
type(target)

#Train test partition of data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(predictors,target,test_size=0.2,random_state=0)


from sklearn.ensemble import GradientBoostingClassifier
boost_clf=GradientBoostingClassifier()
boost_clf.fit(x_train,y_train)

from sklearn.metrics import accuracy_score,confusion_matrix

confusion_matrix(y_test,boost_clf.predict(x_test))
accuracy_score(y_test,boost_clf.predict(x_test))

#Hyperparameters
from sklearn.ensemble import GradientBoostingClassifier
boost_hyper=GradientBoostingClassifier(n_estimators=1000,learning_rate=0.02,max_depth=1)

boost_hyper.fit(x_train,y_train)

from sklearn.metrics import accuracy_score,confusion_matrix

confusion_matrix(y_test,boost_hyper.predict(x_test))
accuracy_score(y_test,boost_hyper.predict(x_test))

#Accuracy of training data

accuracy_score(y_train,boost_hyper.predict(x_train))
#It is giving accuracy of 76% at training and at testing it is 61%
#so model is overfit


