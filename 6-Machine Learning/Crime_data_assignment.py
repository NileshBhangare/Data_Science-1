# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 20:18:45 2023

@author: Nilesh
"""

import pandas as pd
import matplotlib.pyplot as plt
cd=pd.read_csv('c:/2-datasets/crime_data.csv')
cd.columns
cd.shape
cd.dtypes
'''
Unnamed: 0     object
Murder        float64
Assault         int64
UrbanPop        int64
Rape          float64
dtype: object
'''
#droping column unnamed
cd.drop(['Unnamed: 0'],axis=1,inplace=True)

#cheking 5 number summary using describe()
cd.describe()

#as all data is quantitive,no need to create dummy variable
#now normalize data

def normalize(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

cd_norm=normalize(cd)
cd_norm.describe()
#check in variable explorer,
#all data is normalize between 0 to 1


#applying for clustering
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
cd1=linkage(cd_norm,method='complete',metric='euclidean')
plt.figure(figsize=(10,5))
plt.title("Crime data in hierarchical clustering")
sch.dendrogram(cd1,leaf_rotation=0,leaf_font_size=10)
plt.show()


#applying agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
crime=AgglomerativeClustering(n_clusters=4,linkage='complete',affinity='euclidean').fit(cd_norm)
crime.labels_
cluster_labels=pd.Series(crime.labels_)
cd_norm['Cluster']=cluster_labels
res=cd_norm.groupby(cd_norm.Cluster).mean()
res

cd_norm.to_csv("Crimes_cluster.csv",encoding='utf-8')
import os
os.getcwd()
