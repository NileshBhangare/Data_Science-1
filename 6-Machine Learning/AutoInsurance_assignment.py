# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 17:17:54 2023

@author: Nilesh
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
ins=pd.read_csv('c:/2-datasets/AutoInsurance.csv')

#Understanding of data

#1.check for no of rows and columns
ins.shape
#it contains 9134 rows and 24 columns

#2.check data type of columns
ins.dtypes
#In this dataset ,there is multiple column having object datatype

#3.check for duplicates
ins.duplicated().sum()
#There is no duplicates data points

#4.check for missing values
ins.isnull().sum()
#From this we get that there is no any missing values i.e null value

#5.check for outlier
plt.figure(figsize=(20,10))
sns.boxplot(ins)
#from boxplot we get two columns having outlier 
#1)Customer life time value 2)Total claim amount

#renaming column names
ins=ins.rename(columns={'Customer Lifetime Value':'Customer_Lifetime_Value',
                    'Effective To Date':'Effective_To_Date','Location Code':'Location_Code',
                    'Marital Status':'Marital_Status','Monthly Premium Auto':'Monthly_Premium_Auto',
                    'Months Since Last Claim' : 'Months_Since_Last_Claim',
                    'Months Since Policy Inception':'Months_Since_Policy_Inception',
                    'Number of Open Complaints':'Number_of_Open_Complaints',
                    'Number of Policies':'Number_of_Policies',
                    'Policy Type':'Policy_Type','Renew Offer Type':'Renew_Offer_Type',
                    'Sales Channel':'Sales_Channel','Total Claim Amount':'Total_Claim_Amount',
                    'Vehicle Class':'Vehicle_Class','Vehicle Size':'Vehicle_Size'})


#Handling outliers by retain technique
#On Customer_Lifetime_Value column
iqr_clv=ins.Customer_Lifetime_Value.quantile(0.75)-ins.Customer_Lifetime_Value.quantile(0.25)
up_lm=ins.Customer_Lifetime_Value.quantile(0.75)+1.5*iqr_clv
low_lm=ins.Customer_Lifetime_Value.quantile(0.25)-1.5*iqr_clv
outlier_replacement1=pd.DataFrame(np.where(ins.Customer_Lifetime_Value>up_lm,up_lm,
                                           np.where(ins.Customer_Lifetime_Value<low_lm,low_lm,ins.Customer_Lifetime_Value)))
sns.boxplot(outlier_replacement1[0])

#On Total Claim Amount column
iqr_tac=ins.Total_Claim_Amount.quantile(0.75)-ins.Total_Claim_Amount.quantile(0.25)
up_lm=ins.Total_Claim_Amount.quantile(0.75)+1.5*iqr_tac
low_lm=ins.Total_Claim_Amount.quantile(0.25)-1.5*iqr_tac
outlier_replacement2=pd.DataFrame(np.where(ins.Total_Claim_Amount>up_lm,up_lm,
                                           np.where(ins.Total_Claim_Amount<low_lm,low_lm,ins.Total_Claim_Amount)))
sns.boxplot(outlier_replacement2[0])

######################################################

#ins1=ins.drop(['Customer','State','Effective To Date'],axis=1)
ins_new=pd.get_dummies(ins)
#Check for corelation between columns by hitmap
correlation_matrix = ins_new.corr()

# Create a heatmap using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
ins.shape #original dataframe contain 24 columns
ins.shape # after dropping un wanted column we get 21 columns
ins_new.shape 
ins_new.dtypes
#creating dummy variable
#after that we get 60 columns

#checking columns of ins_new dataframe
ins_new.columns
#Now we have Gender ,response columns having only two possible 
#values so we need to drop one column and make only one column 
#after dummy variable
#as we know if we have n values for each columns then after creating dummy variables
#we should have n-1 columns.

#droping that extra columns
ins_new.drop(['Response_No','Coverage_Basic','Education_Bachelor','EmploymentStatus_Disabled','Gender_F','Location Code_Rural','Marital Status_Divorced','Policy Type_Corporate Auto','Renew Offer Type_Offer4','Sales Channel_Agent','Vehicle Class_Luxury Car'],axis=1,inplace=True)
ins_new.columns

#renaming columns
ins_new.rename(columns={'Response_Yes':'Response_No','Gender_M':'Gender'})

#5 number summery
des=ins_new.describe()

#normalizing data
#between 0 to 1
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

ins_norm=norm_fun(ins_new.iloc[:,:])
d=ins_norm.describe()


#Clustering
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage function gives us hierarchical or agglomerative clustering
#ref the help for linkage
b=linkage(ins_norm,method='complete',metric='euclidean')
plt.figure(figsize=(20,17))
plt.title('Hierarchical clustering dendrogram')
sch.dendrogram(b,leaf_rotation=0,leaf_font_size=10)
plt.show()


#using agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=5,linkage='complete',affinity='euclidean').fit(ins_norm)
#apply labels to the clusters
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
#Assign this series to univ dataframe as column and name the column as 'clust'
ins_norm['clust']=cluster_labels
res=ins_norm.groupby(ins_norm.clust).mean()

#copying dataframe to insurance file
ins_norm.to_csv("Insurance.csv",encoding='utf-8')
import os
os.getcwd()
