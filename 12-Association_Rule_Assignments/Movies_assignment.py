# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 13:13:15 2023

@author:  Nilesh
"""

'''************************ Movies.csv ***********************
Bussiness Objective :
    Maximize: Increase User Satisfaction.
    Minimize : Minimizing rate of recommending wrong movies
    
    Business Constraints: Providing currently trending movie
        
'''
'''*****************************Data Dictionary******************

Name of Feature           Description(Genre)                       Type                            Relevance
Sixth Sense                Thriller                     (Categorical)Quantitative                   Relevant
Gladiator                   Drama                       (Categorical)Quantitative                   Relevant
LOTR1                       Adventure                   (Categorical)Quantitative                   Relevant           
Harry Potter1               Fantasy                     (Categorical)Quantitative                   Relevant
Patriot                     Epic                        (Categorical)Quantitative                   Relevant
LOTR2                      Adventure                    (Categorical)Quantitative                   Relevant
Harry Potter2              Fantasy                      (Categorical)Quantitative                   Relevant
LOTR                       Adventure                    (Categorical)Quantitative                   Relevant
Braveheart                 Drama                        (Categorical)Quantitative                   Relevant
Green Mile                Drama,Fantas                  (Categorical)Quantitative                   Relevant
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
mv=pd.read_csv('c:/2-datasets/my_movies.csv')
mv.columns
#It have 10 columns : [Sixth sense,Gladiator,LOTR1,Harry Potter1,Patriot,LOTR2,Harry Potter2,LOTR,Brave Heart,Green Mile]

mv.shape
#It have 10 rows and 10 columns

mv.dtypes
#all features data points are of integer type having value 0 and 1

'''***************************EDA***************************'''
mv.isnull().sum()
#It does not contain any null values

mv.duplicated()
#It contain duplicate data,but we can not delete it as we have only limited number of data

mv.describe()
#From describe, mean of all features is between 0 to 1
# minimum and maximum of all feature is 0 and 1 respectively.

    

''' Checking for oulier '''

sns.boxplot(mv['Sixth Sense'])
#There is no oulier in sixth sense

sns.boxplot(mv['Gladiator'])
#There is no outlier in gladitor

sns.boxplot(mv['LOTR1'])
#There is no outlier in LOTR1

sns.boxplot(mv['Harry Potter1'])
#There is no outlier in Harry Potter1

sns.boxplot(mv['Patriot'])
#there is no outlier in Patriot

sns.boxplot(mv['LOTR2'])
#There is no outlier in LOTR2

sns.boxplot(mv['Harry Potter2'])
#There is no outlier in Harry Potter2

sns.boxplot(mv['LOTR'])
#There is no outlier in LOTR

sns.boxplot(mv['Braveheart'])
#There is no outlier in braveheart

sns.boxplot(mv['Green Mile'])
#there is no outlier in green mile

###################################################################
#Displaying correlation using heatmap

correlation_matrix=mv.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

#From heatmap ,we can take inference that the movies that having same genre are correlated to each other

###################################################################

'''********************************Clustering***************************'''

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(mv,method='complete',metric='Euclidean')
plt.figure(figsize=(15,8))
plt.title('Hierarchical clustering Dendogram')
plt.xlabel('Movies')
plt.ylabel('Distance')

#Dendogram
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()

#Now applying clustering on dataset
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters=3,
                                     linkage='complete',
                                     affinity='euclidean').fit(mv)



