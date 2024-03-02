# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:09:18 2024

@author: Nilesh
"""

'''
A marketing organization outsources thier back off operations to three different suppliers.
The contracts are up for renewal and the CMO wants to determine whether they should renew 
contracts with all suppliers or any specific supplier.CMO wants to renew contracts of 
suppliers with the least transaction time.CMO will renew all contracts if the performance 
of all suppliers is similar.
'''
import pandas as pd
import scipy
from scipy import stats
import numpy as np
import statsmodels.stats.descriptivestats as sd 
import statsmodels.stats.weightstats as stests
con_renewal=pd.read_excel('C:/2-datasets/hypothesis_datasets/ContractRenewal_Data(unstacked).xlsx')
con_renewal
con_renewal.columns='SupplierA','SupplierB','SupplierC'
#H0: All the 3 suppliers have equal mean transaction time
#H1: All the 3 suppliers have not equal mean transaction time
#Normality Test
stats.shapiro(con_renewal.SupplierA)
#pvalue=0.89 >0.005 SupplierA is normal
stats.shapiro(con_renewal.SupplierB)
#pvalue= 0.64>0.005 SupplierB is normal
stats.shapiro(con_renewal.SupplierC)
#pvalue=0.57>0.005 SupplierC is normal

#Varience Test
help(scipy.stats.levene)
#All 3 suppliers are being checked for variences
scipy.stats.levene(con_renewal.SupplierA,con_renewal.SupplierB,con_renewal.SupplierC)
#The levene test tests the null hypothesis
#that all input samples are from populations with equal variences
#pvalue=0.777 >0.005 ,p is high null fly
#H0: all input samples are from populations with equal variences

#One way ANOVA
f,p=stats.f_oneway(con_renewal.SupplierA,con_renewal.SupplierB,con_renewal.SupplierC)

#p value
p # p is high null fly
#pvalue=0.10 >0.05 H0 is accepted