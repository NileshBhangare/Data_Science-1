# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:53:15 2024

@author: Nilesh
"""

#Multiple correlation regression analysis
import pandas as pd
import numpy as np
import seaborn as sns
cars=pd.read_csv('c:/23-Regression/cars.csv')
#Exploratory data analysis
#1.Measure the central tendancy
#2.measure the dispersion(SD)
#3.Third moment business decision(skewness)
#4.Fourth moment business decision(kurtosis)
cars.describe()
#Graphical representations
import matplotlib.pyplot as plt
plt.bar(height=cars.HP,x=np.arange(1,82,1))
sns.distplot(cars.HP)
#data is right skewed
plt.boxplot(cars.HP)
#There are several outliers in HP column
#similar operations are expected for other three column
sns.distplot(cars.MPG)
#data is slightly left skewed
plt.boxplot(cars.MPG)
#There are no outliers
sns.distplot(cars.VOL)
#data is slightly left distributed
plt.boxplot(cars.VOL)
#there are several outliers
sns.distplot(cars.WT)
#data is left skewed
plt.boxplot(cars.WT)
#There are several outliers
sns.distplot(cars.SP)
#data is right skewed
plt.boxplot(cars.SP)
#There are several outliers
#Now let us plot joint plot,joint plot to show scatter diagram
#histogram
import seaborn as sns
sns.jointplot(x=cars['HP'],y=cars['MPG'])

#now let us plot count plot
plt.figure(1,figsize=(16,10))
sns.countplot(cars['HP'])
#Count plot shows how many times the each value occured in dataset
#92 HP value occured 7 times

#QQ plot
from scipy import stats
import pylab
stats.probplot(cars.MPG,dist='norm',plot=pylab)
plt.show()
#MPG data is normally distributed
#There are 10 scatter plot need to be plotted,one by one
#to plot,so we can use pair plots
import seaborn as sns
sns.pairplot(cars.iloc[:,:])
'''
write linearity ,direction and strength
HP -> SP:
HP -> VOL:
HP -> WT:
HP-> MPG:
SP->HP:
SP->VOL:
SP->MPG:
SP->WT:
WT->SP:
WT->VOL:
WT->MPG:
'''
#You can check the collinearity problem between the input WT and VOL
#you can check plot between SP and HP,they are strongly correlated
#same way you can check WT and VOL ,it is also strongle correlated

#now let us check r value between var
cars.corr()
#Sp and Hp r value is 0.97 and same way
#WT and VOL is 0.99  which is greater

#Now although we observed strongly correlated pairs,still we go for linear regression
import statsmodels.formula.api as smf
ml1=smf.ols('MPG~WT+SP+VOL+HP',data=cars).fit()
ml1.summary()
#R square value is 0.771 <0.85
#p values of Wt and VOL is 0.814 and 0.556 which is very high and greater than 0.05
#means we need to ignore WT and VOL columns
#or delete.instead deleting 81 entries,
#let us check ow wise outliers
#identifying is there any influential value
#To check you can use influatial index
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
#76 is the value which has got outliers
#go to data frame and check 76th entry
#let us delete that entry
cars_new=cars.drop(cars.index[[76]])

#again apply regression to cars_new
ml_new=smf.ols('MPG~WT+VOL+HP+SP',data=cars_new).fit()
ml_new.summary()
#R square value is 0.819 but p value are same ,hence not 
#now next option is to delete the column but
#question is which column is to be deleted?
#we already checked correlation factor r
#VOL has got -0.529 and for WT got =0.526
#WT is less hence can be deleted

#another approach is to check the collinearity
#R square is giving that value
#we will have to apply regression w.r.t x1 and input
#as x2,x3 and x4 and so fourth
rsq_hp=smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp=1/(1-rsq_hp)
vif_hp
#VIF is varience influential factor,calculating vif helps to check collinearity
rsq_wt=smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared
vif_wt=1/(1-rsq_wt)
vif_wt

rsq_vol=smf.ols('VOL~HP+WT+SP',data=cars).fit().rsquared
vif_vol=1/(1-rsq_vol)
vif_vol

rsq_sp=smf.ols('SP~HP+WT+VOL',data=cars).fit().rsquared
vif_sp=1/(1-rsq_sp)
vif_sp

#vif_wt=639.53 and vif_vol=638.80 hence vif_wt
#is greater,thumb rule is vif should not be greater than 10

#now let us drop WT and apply correlation to remailing three inputs
final_model=smf.ols('MPG~HP+VOL+SP',data=cars).fit()
final_model.summary()
#Now R square value is 0.77 <0.85 
#and p value of all features is less than 0.05 so this model is selected
final_model1=smf.ols('MPG~HP+VOL+SP',data=cars_new).fit()
final_model1.summary()
#R square =0.81
#all p values are less than 0.05


#prediction
pred=final_model.predict(cars)

#QQ plot
res=final_model.resid
sm.qqplot(res)
plt.show()
#This qq plot is on residual which is obtained on training data
#errors are obtained on test data
stats.probplot(res,dist='norm',plot=pylab)
plt.show()

#let us plot the residual plot,which takes the residuals
# and the data
sns.residplot(x=pred,y=cars.MPG,lowess=True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.show()
#residual plots are used to check whether the errors

#spliting data into train and test
from sklearn.model_selection import train_test_split
cars_train,cars_test=train_test_split(cars,test_size=0.2)
model_train=smf.ols('MPG~VOL+SP+HP',data=cars_train).fit()
model_train.summary()
test_pred=model_train.predict(cars_test)
#test errors
test_errors=test_pred.cars_test.MPG
test_rmse=np.sqrt(np.mean(test_errors*test_errors))
test_rmse