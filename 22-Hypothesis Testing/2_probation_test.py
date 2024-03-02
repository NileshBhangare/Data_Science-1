# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:06:29 2024

@author: Nilesh
"""
'''
Johnnie Talkers soft drinks division sales manager has been planning to launch a new sales 
incentive program for theis sales executives.The sales executives felt that adults(>40yr)
wont buy,children will hence requested sales manager not to launch the program.Analyze the
data and determine whether there is evidence at 5% significance level to support the hypothesis.
'''
import pandas as pd
import scipy
from scipy import stats
import numpy as np
import statsmodels.stats.descriptivestats as sd 
import statsmodels.stats.weightstats as stests
two_prop_test=pd.read_excel('C:/2-datasets/hypothesis_datasets/JohnyTalkers.xlsx')
#H0: probation A =Probation B (check p value)
#H1: Probation A not =Probation B (if pvalue <alpha,we reject H0)

from statsmodels.stats.proportion import proportions_ztest
tab1=two_prop_test.Person.value_counts()
tab1
tab2=two_prop_test.Drinks.value_counts()
tab2

#Crosstable 
pd.crosstab(two_prop_test.Person,two_prop_test.Drinks)
count=np.array([58,152])
nobs=np.array([480,740])

stats,pval=proportions_ztest(count, nobs,alternative='two-sided')
print(pval) #pvalue=0.000

stats,pval=proportions_ztest(count, nobs,alternative='larger')
print(pval)
#pvalue= 0.999