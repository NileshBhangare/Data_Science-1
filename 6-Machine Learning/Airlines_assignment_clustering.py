# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 20:45:03 2023

@author: Nilesh
"""

import pandas as pd
import matplotlib.pyplot as plt
air=pd.read_excel('c:/2-datasets/EastWestAirlines.xlsx')
air.dtypes
air.shape
air.columns
#as we dont want id so droping ID column
air.drop(['ID#'],axis=1,inplace=True)

#now check 5 number summary by describe method
al=air.describe()
#normalizing data
def normalize(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

air_norm=normalize(air)
al_des=air_norm.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
airline=linkage(air_norm,method='complete',metric='euclidean')
plt.figure(figsize=(15,8))
plt.title("Clustering on Airline Data")
sch.dendrogram(airline,leaf_rotation=0,leaf_font_size=10)
plt.show()


#applying agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
air1=AgglomerativeClustering(n_clusters=10,linkage='complete',affinity='euclidean').fit(air_norm)
air1.labels_
cluster_labels=pd.Series(air1.labels_)
air_norm['Cluster']=cluster_labels
r1=air_norm.groupby(air_norm.Cluster).mean()
r1

air_norm.to_csv("Airlines_cluster.csv",encoding='utf-8')
import os
os.getcwd()
