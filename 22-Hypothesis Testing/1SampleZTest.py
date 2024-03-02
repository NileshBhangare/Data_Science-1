# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:19:15 2024

@author: Nilesh
"""

import pandas as pd
import scipy
from scipy import stats
import numpy as np
import statsmodels.stats.descriptivestats as sd 
import statsmodels.stats.weightstats as stests

fabric=pd.read_csv('c:/2-datasets/hypothesis_datasets/Fabric_data.csv')

#Hypothesis
#H0 : average length of fabric is less than or equal to 150
#H1 : average length of fabric is greater than 150 

#Calculating normality test
print(stats.shapiro(fabric))
#0.1460 >0.005 H0 True
#Calculating the mean
np.mean(fabric)

#ztest
#Parameters in ztest,value is mean of data
ztest,pval=stests.ztest(fabric,x2=None,value=150)

print(float(pval))

#p-value =7.156e-06 <0.05 so p is low ,null will go
