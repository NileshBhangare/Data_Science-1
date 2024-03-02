# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:32:47 2024

@author: Nilesh
"""

import pandas as pd
import scipy
from scipy import stats
import numpy as np
import statsmodels.stats.descriptivestats as sd 
import statsmodels.stats.weightstats as stests

fuel=pd.read_csv("c:/2-datasets/hypothesis_datasets/mann_whitney_additive.csv")
fuel

fuel.columns="Without_additive","With_additive"

#Normality Test
print(stats.shapiro(fuel.Without_additive))
#p is high,null fly
#H0 : data is normal
print(stats.shapiro(fuel.With_additive))
# p is low,null go
# data is not normal

#When two samples are not normal then mannwhitney test
#Non parametric test case
#Mann-whitney Test
scipy.stats.mannwhitneyu(fuel.Without_additive, fuel.With_additive)
#p value=0.4457 >0.05 so p high null fly
#H0: fuel additive does not impact the performance
