# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 09:19:10 2023

@author: Nilesh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#let us try to understand first how k means works for two
#dimentional data
#for that,generate random numbers in range of 0 to 1
#and with uniform probability of 1/50
X=np.random.uniform(0,1,50)
Y=np.random.uniform(0,1,50)
#create a empty dataframe with 0 rows and 2 columns
df_xy=pd.DataFrame(columns=['X','Y'])
#assign the values of X and Y to these columns
df_xy.X=X
df_xy.Y=Y
df_xy.plot(x='X',y='Y',kind="scatter")
model1=KMeans(n_clusters=3).fit(df_xy)

'''
With the data X and Y apply KMEans model,
generate scatter plot
with scale/font=10

cmap=plt.cm.coolwarm:cool color combination
'''
model1.labels_
df_xy.plot(x='X',y='Y',c=model1.labels_,kind='scatter',s=10,
           cmap=plt.cm.coolwarm)
Univ1=pd.read_excel('c:/2-datasets/University_Clustering.xlsx')
Univ1.describe()
univ=Univ1.drop(['State'],axis=1)
#we know that there is scale differance among the column,
#which we have to solve either by normalization or standardization
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

#now apply this normalization function to univ dataframe for all the rows

df_norm=norm_fun(univ.iloc[:,1:])

'''
what will be ideal cluster number,will it be 1,2 or 3
'''
TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    
TWSS.append(kmeans.inertia_) # total within the sum of square

'''
Kmeans inertia,also known as sum of squares errors
(SSE) ,calcualtes the sum of distances of all points within a cluster
from the centroid of the point,It is the difference between 
the observed value and predicted value
'''
TWSS
plt.plot(k,TWSS,'ro-')
plt.xlabel("No of clusters")
plt.ylabel("Total_within_SS")

'''
How to slect value of k from elbow curve

'''
model=KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
mb=pd.Series(model.labels_)
univ['clust']=mb
univ.head()
univ=univ.iloc[:,[7,0,1,2,3,4,5,6]]
univ
univ.iloc[:,2:8].groupby(univ.clust).mean()
univ.to_csv("kmeans_university.csv",encoding='utf-8')
import os
os.getcwd()


    
                 