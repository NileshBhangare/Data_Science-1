# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:18:11 2024

@author: Nilesh
"""

import pandas as pd
import scipy
from scipy import stats
import numpy as np
import statsmodels.stats.descriptivestats as sd 
import statsmodels.stats.weightstats as stests
Bahaman=pd.read_excel('C:/2-datasets/hypothesis_datasets/Bahaman.xlsx')
Bahaman

count=pd.crosstab(Bahaman['Defective'], Bahaman['Country'])
count
Chisquares_results=scipy.stats.chi2_contingency(count)
Chi_square=[['Test Statistic','p-value'],[Chisquares_results[0],Chisquares_results[1]]]
Chi_square

'''
You use chi2_contingency when you want to test
whether two (or more) groups have the same distribution

'''
#H0:Null hypothesis:the two groups have no significant difference
#since p=0.63>0.05 Hence H0 is true