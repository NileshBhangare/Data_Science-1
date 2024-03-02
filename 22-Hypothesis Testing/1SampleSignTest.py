# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 09:36:28 2024

@author: Nilesh
"""

import pandas as pd
import scipy
from scipy import stats

import statsmodels.stats.descriptivestats as sd 
import statsmodels.stats.weightstats
#1 sample sign test
#for given datasey check whether scores are equal or less than 80
#H0: scores are either equal or less then 80
#H1: scores are not equal and greater then 80
#Whenever there is single sample and data is not normal
marks=pd.read_csv('c:/2-datasets/hypothesis_datasets/Signtest.csv')
#Normal QQ plot
import pylab
stats.probplot(marks.Scores,dist='norm',plot=pylab)
#Data is not normal
#H0: data is normal
#H1: data is not normal
stats.shapiro(marks.Scores)
#p_value is 0.024>0.005,p is high null fly
#Decision: data is not normal
###########################
#let us check the distribution of the data
marks.Scores.describe()
#1 sample sign test
sd.sign_test(marks.Scores,mu0=marks.Scores.mean())
#p_value 0.82> 0.005,p is high null fly
#Decision: 
    #H0 : scores are either equal or less than 80


