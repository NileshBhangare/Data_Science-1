# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 22:10:50 2023

@author: Nilesh
"""
import pandas as pd
import matplotlib.pyplot as plt
t=pd.read_excel('c:/2-datasets/Telco_customer_churn.xlsx')
t.shape
t.columns
t.dtypes
#we have three columns that are not giving much information
#and contributing in result
#So we are droping customer_id ,count and quarter
t.drop(['Customer ID','Count','Quarter'],axis=1,inplace=True)
t1=pd.get_dummies(t)
t1.columns

#droping unneccessary columns
t1.drop(['Referred a Friend_No','Offer_None','Phone Service_No','Multiple Lines_No','Internet Service_No','Internet Type_Cable','Online Security_No','Online Backup_No','Device Protection Plan_No','Premium Tech Support_No','Streaming TV_No','Streaming Movies_No','Streaming Music_No','Unlimited Data_No','Contract_Two Year','Paperless Billing_No','Payment Method_Bank Withdrawal'],axis=1,inplace=True)
#renaming columns
t1.rename(columns={'Referred a Friend_Yes':'Referred a friend','Streaming Music_Yes':'Streaming music','Phone Service_Yes':'Phone Service','Multiple Lines_Yes':'Multiple Lines','Internet Service_Yes':'Internet Service','Online Security_Yes':'Online Security','Online Backup_Yes':'Online Backup','Device Protection Plan_Yes':'Device Production Plan','Premium Tech Support_Yes':'Premium Tech Support','Streaming TV_Yes':'Streaming TV','Streaming Movies_Yes':'Streaming Movies','Unlimited Data_Yes':'Unlimited Data','Paperless Billing_Yes':'Paperless Billing'},inplace=True)
t1.columns

d1=t1.describe()

#normalizing data
def normalize(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

t1_norm=normalize(t1)
d2=t1_norm.describe()

#clustering
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
tc=linkage(t1_norm,method='complete',metric='euclidean')
plt.figure(figsize=(17,10))
plt.title("Churn Decision clustering")
sch.dendrogram(tc,leaf_rotation=0,leaf_font_size=10)
plt.show()


from sklearn.cluster import AgglomerativeClustering
tc1=AgglomerativeClustering(n_clusters=10,linkage='complete',affinity='euclidean').fit(t1_norm)
tc1.labels_
cluster_labels=pd.Series(tc1.labels_)
t1_norm['Cluster']=cluster_labels
grp=t1_norm.groupby(t1_norm.Cluster).mean()

t1_norm.to_csv('Churn.csv',encoding='utf-8')
import os
os.getcwd()
