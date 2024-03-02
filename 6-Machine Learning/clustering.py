# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:14:26 2023

@author: Nilesh
"""

import pandas as pd
import matplotlib.pyplot as plt
Univ1=pd.read_excel('c:/2-datasets/University_Clustering.xlsx')
a=Univ1.describe()
#we have one column state which does not have much use
univ=Univ1.drop(['State'],axis=1)

#We know the scale difference among the columns is very high 
#so we need to normalize this data into the range of 0 and 1 
#using normalization 
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

#Now apply this normaalization function to univ dataframe
#for all the rows and column from 1 to until end
#since 0th column has university name hence skipped
df_norm=norm_fun(univ.iloc[:,1:])
#Now you can check df_norm dataframe which is scaled
#between 0 and 1
#you can apply describe function to new data frame
b=df_norm.describe()

#Before you apply clustering you need to plot dendrogram first
#Now to create dendrogram we need to measure distance
#we have to impport linkage

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage function gives us hierarchical or addlomerative clustering
#ref the help for linkage
z=linkage(df_norm,method='complete',metric='euclidean')
plt.figure(figsize=(15,8))
plt.title('Hierarchical clustering dendrogram')
plt.xlabel("Index")
plt.ylabel("Distance")
#ref help for dendrogram
#sch.dendrogram(z)
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()

#applying agglomerative clustering chossing 3 as clusters
#from dendrogram
#whatever has been displayed in dendrogram is not clustering
#it is just showing number of possible clusters

from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(df_norm)
#apply labels to the clusters
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
#Assign this series to univ dataframe as column and name the column as 'clust'
univ['clust']=cluster_labels
new_univ=univ.iloc[:,[7,1,2,3,4,5,6]]
new_univ.iloc[:,2:].groupby(new_univ.clust).mean()
#from the output cluster 2 has got highest top 10
#lowest accpet ration,best faculty ration and highest expenses
#highest graduates ratio
new_univ.to_csv("University.csv",encoding='utf-8')
import os
os.getcwd()


