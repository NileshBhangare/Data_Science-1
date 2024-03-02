# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:00:14 2024

@author: Nilesh
"""
'''
A finacial analyst at a financial institute wants to evaluate a recent credit card promotiion.
After this promotion,500 cardholders were randomly selected.Half received an ad promoting a full
waiver of interest rate on purchases made over the next three months,and half received a standard
crismas ad.Did the ad promoting full ineterest rate waiver,increase purchases?

'''
import pandas as pd
import scipy
from scipy import stats
import numpy as np
import statsmodels.stats.descriptivestats as sd 
import statsmodels.stats.weightstats as stests

#load the data
prom=pd.read_excel('c:/2-datasets/hypothesis_datasets/Promotion.xlsx')
prom
#H0: InterestRateWaiver < StandardPromotion
#H1: InterestRateWaiver > StandardPromotion

prom.columns="InterestRateWaiver","StandardPromotion"

#Normality test
stats.shapiro(prom.InterestRateWaiver)
print(stats.shapiro(prom.StandardPromotion))

#data is normal

#varience test
help(scipy.stats.levene)
#H0:Both columns have equal varience
#H1: Both columns does not have equal varience
scipy.stats.levene(prom.InterestRateWaiver,prom.StandardPromotion)
#p_value=0.287 > 0.05 so p is high null fly => equal variences

#2 sample T test
scipy.stats.ttest_ind(prom.InterestRateWaiver,prom.StandardPromotion)
help(scipy.stats.ttest_ind)
#H0: equal means
#H1:unequal means
#p -value= 0.024 <0.05 so p is low null go
#H1: InterestRateWaiver > StandardPromotion => decision