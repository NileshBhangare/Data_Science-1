import pandas as pd
import numpy as np
wcat=pd.read_csv('c:/2-datasets/wc-at.csv')
#Exploratory data analysis
#1-Measure the central tendancy
#2-measures of dispersion
#3-Third moment business decision
#4-Fourth moment business decision
wcat.info()
wcat.describe()
#Graphical Representation
import matplotlib.pyplot as plt
plt.bar(height=wcat.AT,x=np.arange(1,110,1))
plt.hist(wcat.AT)
plt.boxplot(wcat.AT)
#data is right skewed
#scatter plot
plt.scatter(x=wcat['Waist'],y=wcat['AT'],color='green')

##now let us check direction using covariance factor
cov_output=np.cov(wcat.Waist,wcat.AT)[0,1]
cov_output
#635.91000

#now let us apply linear regression model
import statsmodels.formula.api as smf
#All machine learning algorithms are implemented using sklearn
#but for this statsmodel package is being used because it gives you all backend
#calculations of bita 0 and bita 1
model=smf.ols('AT~Waist',data=wcat).fit()
model.summary()
#OLS helps to find best fit odel,which causes least square error
#First you check R squared value=0.670,if R square=0.8 means that model
#is not best fit,if R-square=0.8 to 0.6 moderate correlation
#Next you check P>|t| =0 ,it means less than alpha,alpha is 0.05
#hence model is accepted

#Regression line
pred1=model.predict(pd.DataFrame(wcat['Waist']))
plt.scatter(wcat.Waist,wcat.AT)
plt.plot(wcat.Waist,pred1,'r')
plt.show()

#Error calculation
res1=wcat.AT-pred1
res1
np.mean(res1)
#it must be zero and here it 10^-14=~0
res_sqr1=res1*res1
mse1=np.mean(res_sqr1)
rmse1=np.sqrt(mse1)
rmse1
#32.7601 lessar the value better the model
#How to improve this model,Transformation of WC
plt.scatter(x=np.log(wcat['Waist']), y=wcat['AT'], color='brown')
#data is linearly scattered ,positive direction
np.corrcoef(np.log(wcat.Waist),wcat.AT)
#r value is 0.82<0.85 hence moderate linearity
model2=smf.ols('AT~np.log(Waist)',data=wcat).fit()
model2.summary()
#First you check R squared value=0.675,if R square=0.8 means that model
#is not best fit,if R-square=0.8 to 0.6 moderate correlation
#Next you check P>|t| =0 ,it means less than alpha,alpha is 0.05
#hence model is accepted
pred2=model2.predict(pd.DataFrame(wcat['Waist']))
plt.scatter(np.log(wcat.Waist),wcat.AT)
plt.plot(np.log(wcat.Waist),pred2,'r')
plt.legend('Predicted line','observed data')
plt.show()

#Error calculation
res2=wcat.AT-pred2
res2
np.mean(res2)
#it must be zero and here it 10^-14=~0
res_sqr2=res2*res2
mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2
#32.4968 error

#no any considerable changes
#again need to improve model by transforming AT
plt.scatter(x=wcat['Waist'], y=np.log(wcat['AT']), color='brown')
np.corrcoef(wcat.Waist,np.log(wcat.AT))
#r value is 0.84<0.85 hence moderate linearity
model3=smf.ols('np.log(AT)~Waist',data=wcat).fit()
model3.summary()
#First you check R squared value=0.707,if R square=0.8 means that model
#is not best fit,if R-square=0.8 to 0.6 moderate correlation
#Next you check P>|t| =0.002 ,it means less than alpha,alpha is 0.05
#hence model is accepted
pred3=model3.predict(pd.DataFrame(wcat['Waist']))
pred3_at=np.exp(pred3)
#here we take exp(pred3) because we taking log(y) i.e log(AT)
#check wcat and pred3_at from variable explorer
#scatter diagram
plt.scatter(wcat.Waist,np.log(wcat.AT))
plt.plot(wcat.Waist,pred3,'r')
plt.legend('Actual Data','Predicted data')
plt.show()

#Error calculation
res3=wcat.AT-pred3_at
res3

#it must be zero and here it 10^-14=~0
res_sqr3=res3*res3

mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3
#38.5290 #not much greater

#need to improve model again
################################################
#using polynomial transformation
#x=waist,x^2=waist*waist,y=log(at)
#here r can not be calculated
model4=smf.ols('np.log(AT)~Waist+I(Waist*Waist)',data=wcat).fit()
#Y is log(AT) and x=Waist
model4.summary()
#R-squared=0.779<0.85,there is scope of improvement
#p=0.0000<0.05 hence acceptable
#bita-0=-7.8241
#bita-1=0.2289
pred4=model4.predict(pd.DataFrame(wcat.Waist))
pred4
pred4_at=np.exp(pred4)
pred4_at
#############################
#Regression line
plt.scatter(wcat.Waist,np.log(wcat.AT))
plt.plot(wcat.Waist,pred4,'r')
plt.legend(['Actual data','predicted data'])
plt.show()
###############################
#error Calculations
res4=wcat.AT-pred4_at
res4

#it must be zero
res_sqr4=res4*res4

mse4=np.mean(res_sqr4)
rmse4=np.sqrt(mse4)
rmse4
#32.24
#Among all the models,model4 is best
############################################
data={'model':pd.Series(['SLR','Log_model','Exp_model','Poly_model'])}
data
table_rmse=pd.DataFrame(data)
table_rmse
#########################################
#we have to generalize the best model
from sklearn.model_selection import train_test_split
train,test=train_test_split(wcat,test_size=0.2)
plt.scatter(train.Waist,np.log(train.AT))
final_model=smf.ols('np.log(AT)~Waist+I(Waist*Waist)',data=wcat).fit()
#Y is log(AT) and X=Waist
final_model.summary()
#R-squared=0.779<0.85,there is scope of improvement
#P=0.00<0.05 hence acceptable
#bita-0=-7.8241
#bita-1=0.2289
test_pred=final_model.predict(pd.DataFrame(test))
test_pred_at=np.exp(test_pred)
test_pred_at
######################
train_pred=final_model.predict(pd.DataFrame(train))

train_pred_at=np.exp(train_pred)
train_pred_at
###########################################
#Evaluation on test data
test_err=test.AT-test_pred_at
test_sqr=test_err*test_err
test_mse=np.mean(test_sqr)
test_rmse=np.sqrt(test_mse)
test_rmse

#Evaluation on train data
train_res=train.AT-train_pred_at
train_sqr=train_res*train_res
train_mse=np.mean(train_sqr)
train_rmse=np.sqrt(train_mse)
train_rmse
#################################
#test_rmse>train_rmse,hence the model is overfit

