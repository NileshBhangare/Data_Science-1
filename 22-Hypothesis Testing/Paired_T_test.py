# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:40:54 2024

@author: Nilesh
"""

import pandas as pd
import scipy
from scipy import stats
import numpy as np
import statsmodels.stats.descriptivestats as sd 
import statsmodels.stats.weightstats as stests

#when two features are normal then paired T test
#A univariate test that tests for a significant difference between 2 

sup=pd.read_csv('c:/2-datasets/hypothesis_datasets/paired2.csv')
sup.describe()

#H0: There is no significant difference between means of supliers of A and B
#H1 : There is significant difference between means of supliers of A and B

#Normality test - Shapiro test
stats.shapiro(sup.SupplierA)
stats.shapiro(sup.SupplierB)
#data is normal

import seaborn as sns
sns.boxplot(data=sup)

#Assuming the external conditions are same for both the samples
#Paired T test
ttest,pval=stats.ttest_rel(sup.SupplierA,sup.SupplierB)
print(pval)
#p val= 0 <0.05 p val is low,null go
# H1: There is significant difference between means of supliers of A and B
