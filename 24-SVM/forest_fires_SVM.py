# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:30:12 2024

@author:  Nilesh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
forest=pd.read_csv('c:/24-SVM/forestfires.csv')

forest.dtypes
forest.columns
forest['month'].astype(str)
###########################################
forest.shape
#It contains total 31 features and 517 records or data points

plt.figure(1,figsize=(16,10))
sns.countplot(forest.month)
#aug and sept has highest value
sns.countplot(forest.day)
#friday,sunday and saturday has highest value

sns.distplot(forest.FFMC)
#data isnormal and slight left skewed
sns.boxplot(forest.FFMC)
#There are several outliers

sns.distplot(forest.DMC )
#data isnormal and slight right skewed
sns.boxplot(forest.DMC)
#There are several outliers

sns.distplot(forest.DC)
#data isnormal and slight left skewed
sns.boxplot(forest.DC)
#There are several outliers

sns.distplot(forest.ISI)
#data isnormal 
sns.boxplot(forest.ISI)
#There are several outliers

sns.distplot(forest.temp)
#data isnormal 
sns.boxplot(forest.temp)
#There are several outliers

sns.distplot(forest.RH)
#data isnormal and slight right skewed
sns.boxplot(forest.RH)
#There are several outliers

sns.distplot(forest.wind)
#data isnormal and slight right skewed
sns.boxplot(forest.wind)
#There are several outliers

sns.distplot(forest.rain )
#data isnormal 
sns.boxplot(forest.rain )
#There are several outliers

sns.distplot(forest.area)
#data isnormal 
sns.boxplot(forest.area)
#There are several outliers

#Now let us check the highest fire in KM
forest.sort_values(by='area',ascending=False).head(5)

highest_fire_area=forest.sort_values(by='area',ascending=False)

plt.figure(figsize=(8,6))

plt.title('Tempreture vs area of fire')
plt.bar(highest_fire_area['temp'],highest_fire_area['area'])

plt.xlabel('Tempreture')
plt.ylabel('Area per km-sq')
plt.show()
#Once the fire starts,almost 1000+ sq areas

#tempreture goes beyond 25 and
#around 750km area is facing temp 30+
#now let us check the highest rain in the forest
highest_rain=forest.sort_values(by='rain',ascending=False)[['month','day','rain']].head(5)

highest_rain
#highest rain observed in the month of aug
#let us check highest and lowest tempreture in month and day
highest_temp=forest.sort_values(by='temp',ascending=False)[['month','day','rain']].head(5)

lowest_temp=forest.sort_values(by='temp',ascending=True)[['month','day','rain']].head(5)

print('Highest Tempreture ',highest_temp)
#Highest temp observerd in month of aug
print('Lowest Tempreture: ',lowest_temp)
#lowest temp observed in month of dec

forest.isna().sum()
#There is no any NA value

############################################
from sklearn.preprocessing import LabelEncoder 
labelencoder=LabelEncoder()
forest.month=labelencoder.fit_transform(forest.month)
forest.day=labelencoder.fit_transform(forest.day)
forest.size_category=labelencoder.fit_transform(forest.size_category)

forest.dtypes
#Now are features are in numeric form

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['month'])
df_t=winsor.fit_transform(forest[['month']])
sns.boxplot(df_t.month)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['FFMC'])
df_t=winsor.fit_transform(forest[['FFMC']])
sns.boxplot(df_t.FFMC)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['DMC'])
df_t=winsor.fit_transform(forest[['DMC']])
sns.boxplot(df_t.DMC)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['DC'])
df_t=winsor.fit_transform(forest[['DC']])
sns.boxplot(df_t.DC)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['ISI'])
df_t=winsor.fit_transform(forest[['ISI']])
sns.boxplot(df_t.ISI)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['temp'])
df_t=winsor.fit_transform(forest[['temp']])
sns.boxplot(df_t.temp)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['RH'])
df_t=winsor.fit_transform(forest[['RH']])
sns.boxplot(df_t.RH)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['wind'])
df_t=winsor.fit_transform(forest[['wind']])
sns.boxplot(df_t.wind)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['rain'])
df_t=winsor.fit_transform(forest[['rain']])
sns.boxplot(df_t.rain)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['area'])
df_t=winsor.fit_transform(forest[['area']])
sns.boxplot(df_t.area)


tc=forest.corr()
tc
fig,ax=plt.subplots()
fig.set_size_inches(200,10)
sns.heatmap(tc,annot=True,cmap='YlGnBu')
#all variables are moderately correlated with size_category

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test=train_test_split(forest,test_size=0.3)
train_x=train.iloc[:,:30]
train_y=train['size_category']
test_x=test.iloc[:,:30]
test_y=test['size_category']
#Kernel linear
model_linear=SVC(kernel='linear')
model_linear.fit(train_x,train_y)
pred_test_linear=model_linear.predict(test_x)
np.mean(pred_test_linear==test_y)
#It gives the accuracy of 98%
#model is overfit
#RBF
model_rbf=SVC(kernel='rbf')
model_rbf.fit(train_x,train_y)
pred_test_rbf=model_rbf.predict(test_x)
np.mean(pred_test_rbf==test_y)
#It gives the accuracy of 71%

#Using gamma value
gamma_value = 1  # You can change this value to your desired gamma

# Kernel linear with specified gamma
model_gamma = SVC(kernel='linear', gamma=gamma_value)
model_gamma.fit(train_x, train_y)
pred_test_gamma = model_gamma.predict(test_x)

# Calculate accuracy
accuracy_gamma = np.mean(pred_test_gamma == test_y)
print("Accuracy with gamma value {}:".format(gamma_value), accuracy_gamma)
