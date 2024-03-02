# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:26:31 2023

@author: Nilesh
"""

#**********************Association Rule Assignemts ****************************
#*********************** Books.csv ********************************************

''' Business objective :
     Maximize : To increase the Sell of books
    Minimize :  
        
    
    Business Constraints:
        Managing demand of customer according to trends in market.
'''

'''*************************Data Dictionary*********************************
Name of feature    Description          Type                                Relevance

ChildBks           Child Books        (Categorical)Quantitative
YouthBks           Youth Books        (Categorical)Quantitative
CookBks            Cooking Books      (Categorical)Quantitative
DoItYBks           Do It books        (Categorical)Quantitative
RefBks             Reference Books    (Categorical)Quantitative
ArtBks             Art Books          (Categorical)Quantitative
GeogBks            Geography Books    (Categorical)Quantitative
ItalCook           Itallian Books     (Categorical)Quantitative
ItalAtlas          Ital At las Books  (Categorical)Quantitative
ItalArt            Italian Art Books  (Categorical)Quantitative
Florence           Folarance Books    (Categorical)Quantitative
    
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
book=pd.read_csv('c:/2-datasets/book.csv')
book
book.columns 
''' It have 11 columns
    ChildBks   
    YouthBks     
    CookBks
    DoItYBks     
    RefBks       
    ArtBks       
    GeogBks      
    ItalCook     
    ItalAtlas    
    ItalArt      
    Florence     
'''          
book.dtypes
#All features are of numerical type having value 0 and 1

book.shape
# The dataset contain 2000 datapoints with 11 features

''' **************************** EDA *******************************'''

des=book.describe()

#From describe, mean of all features is between 0 to 1
# minimum and maximum of all feature is 0 and 1 respectively.


book.isnull().sum()
#There is no any null value

''' ************************** Checking for outliers ******************* '''
#On first feature ChildBks

sns.boxplot(book['ChildBks'])
#There is no outliers for this feature.
#All datapoints are symmentrycally nrmally distributed

#On YouthBks 
sns.boxplot(book['YouthBks'])
#There is only one outlier

#Outlier treatment
youth_iqr=book['YouthBks'].quantile(0.75)-book['YouthBks'].quantile(0.25)
youth_q1=book['YouthBks'].quantile(0.25)
youth_q3=book['YouthBks'].quantile(0.75)

#Finding lower and upper limit
'''
lower_limit=q1-1.5*iqr
upper_limit=q3+1.5*iqr
'''
youth_l_limit=youth_q1-1.5*youth_iqr 
youth_u_limit=youth_q3+1.5*youth_iqr

book['YouthBks']=np.where(book['YouthBks']>youth_u_limit,youth_u_limit,np.where(book['YouthBks']<youth_l_limit,youth_l_limit,book['YouthBks']))
sns.boxplot(book['YouthBks'])

#On CookBks column

sns.boxplot(book['CookBks'])
#There is no outlier 

#On DoItYBks column
sns.boxplot(book['DoItYBks'])
#There is no outlier

#On RefBks column
sns.boxplot(book['RefBks']) 
#Outlier Treatment
ref_iqr=book['RefBks'].quantile(0.75)-book['RefBks'].quantile(0.25)
ref_q1=book['RefBks'].quantile(0.25)
ref_q3=book['RefBks'].quantile(0.75)

ref_l_limit=ref_q1-1.5*ref_iqr
ref_u_limit=ref_q3+1.5*ref_iqr

book['RefBks']=np.where(book['RefBks']>ref_u_limit,ref_u_limit,np.where(book['RefBks']<ref_l_limit,ref_l_limit,book['RefBks']))
sns.boxplot(book['RefBks'])

#On ArtBks column
sns.boxplot(book['ArtBks']) 
#There is outlier

#Outlier treatment
art_iqr=book['ArtBks'].quantile(0.75)-book['ArtBks'].quantile(0.25)
art_q1=book['ArtBks'].quantile(0.25)
art_q3=book['ArtBks'].quantile(0.75)

art_l_limit=art_q1-1.5*art_iqr
art_u_limit=art_q3+1.5*art_iqr
book['ArtBks']=np.where(book['ArtBks']>art_u_limit,art_u_limit,np.where(book['ArtBks']<art_l_limit,art_l_limit,book['ArtBks']))
sns.boxplot(book['ArtBks'])
#Now there is no any outlier

#On GeogBks column
sns.boxplot(book['GeogBks'])
#There is no any outlier

#On ItalCook column
sns.boxplot(book['ItalCook'])
#There is outlier

#Outlier Treatment
iqr=book['ItalCook'].quantile(0.75)-book['ItalCook'].quantile(0.25)
q1=book['ItalCook'].quantile(0.25)
q3=book['ItalCook'].quantile(0.75)

l_limit=q1-1.5*iqr
u_limit=q3-1.5*iqr
book['ItalCook']=np.where(book['ItalCook']>u_limit,u_limit,np.where(book['ItalCook']<l_limit,l_limit,book['ItalCook']))
sns.boxplot(book['ItalCook'])

#Now there is no any outlier

#On ItalAtlas column
sns.boxplot(book['ItalAtlas'])
#There is outlier
iqr=book['ItalAtlas'].quantile(0.75)-book['ItalAtlas'].quantile(0.25)
q1=book['ItalAtlas'].quantile(0.25)
q3=book['ItalAtlas'].quantile(0.75)

l_limit=q1-1.5*iqr
u_limit=q3-1.5*iqr
book['ItalAtlas']=np.where(book['ItalAtlas']>u_limit,u_limit,np.where(book['ItalAtlas']<l_limit,l_limit,book['ItalAtlas']))
sns.boxplot(book['ItalAtlas'])
#Now all outlier is removed

#On ItalArt column
sns.boxplot(book['ItalArt'])
#There is outlier
iqr=book['ItalArt'].quantile(0.75)-book['ItalArt'].quantile(0.25)
q1=book['ItalArt'].quantile(0.25)
q3=book['ItalArt'].quantile(0.75)

l_limit=q1-1.5*iqr
u_limit=q3-1.5*iqr
book['ItalArt']=np.where(book['ItalArt']>u_limit,u_limit,np.where(book['ItalArt']<l_limit,l_limit,book['ItalArt']))
sns.boxplot(book['ItalArt'])
#Now all outliers are removed

#On Florance column
sns.boxplot(book['Florence'])
#There is outlier
iqr=book['Florence'].quantile(0.75)-book['Florence'].quantile(0.25)
q1=book['Florence'].quantile(0.25)
q3=book['Florence'].quantile(0.75)

l_limit=q1-1.5*iqr
u_limit=q3-1.5*iqr
book['Florence']=np.where(book['Florence']>u_limit,u_limit,np.where(book['Florence']<l_limit,l_limit,book['Florence']))
sns.boxplot(book['Florence'])
#now all outlier are removed


#Displaying correlation using heatmap
correlation_matrix = book.corr()

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

# from heapmap, we understand that all features are corelated each other 
#so no need to drop any feature


''' ***************************** Clustring ******************************'''
#Dendogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(book,method='complete',metric='euclidean')
plt.figure(figsize=(15,8))
plt.title('Hierarchical clustering dendrogram')
plt.xlabel('index')
plt.ylabel('distance')
#dendrogram
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()
#now apply clustering 
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters=3,
                                     linkage='complete',
                                     affinity='euclidean').fit(book)
#apply labels to clusters
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
#assign this series to  dataframe as column
book['cluster'] = cluster_labels


#bookNew = book.iloc[:,[-1,0,1,2,3,4,5,6,7,8,9,10,11]]
result=book.iloc[:, 2:].groupby(book['cluster']).mean()

book.to_csv("bookNew.csv",encoding='utf-8')
book.cluster.value_counts()
import os
os.getcwd()


''' *******************************K means Clustering ***********************************'''
#for this we will used normalized data set book

from sklearn.cluster import KMeans
#total sum of squares
TWSS = []

#initially we will find the ideal cluster number using elbow curve

k = list(range(2,8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(book)
    TWSS.append(kmeans.inertia_)
  
TWSS
'''
1550.5069257514326,
 1079.2483102544325,
 815.1685947628789,
 669.1130754513432,
 520.9328703047712,
 352.53830308786763
'''
def find_cluster_number(TWSS):
    diff =[]
    for i in range(0,len(TWSS)-1):
        d = TWSS[i]-TWSS[i+1]
        diff.append(d)
    max = 0
    k =0
    for i in range(0,len(diff)):
        if max<diff[i]:
            max = diff[i]
            k = i+6
    return k

k = find_cluster_number(TWSS)
print("Cluster number is = ",k)
plt.plot(k,TWSS,'-')
plt.xlabel('No of clusters')
plt.ylabel('Total_within_SS')

model = KMeans(n_clusters=k)
model.fit(book)
model.labels_
mb = pd.Series(model.labels_)
book['clusters'] = mb
book.head()
book.shape
book.columns
book = book.iloc[:,[-1,0,1,2,3,4,5,6,7,8,9,10]]
book
book.iloc[:,2:11].groupby(book.clusters).mean()
book.to_csv('bookNew_kmean.csv')
import os
os.getcwd()


''' ****************************Association Rule On Book.csv**********************'''

from collections import Counter
item_frequencies=Counter(book.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]])
#item_frequencies is basically dictionary having 
#x[0] as key and x[1]=values
#we want to access values and sort based on the count that occured in it.
#it will show the count of each item purchased in every transaction
#Now let us sort these frequencies in ascending order
item_frequencies=sorted(item_frequencies.items(),key=lambda x:x[1])
#when we execute this,item frequencies will be in sorted form,in the form
# item name with count
#Let us seperate out items and their count
items=list(reversed([i[0] for i in item_frequencies]))
#This is list comprehension for each item in item frequencies access the key
#here you will get itme list
frequencies=list(reversed([i[1] for i in item_frequencies]))
#here you will get count of purchase of each item

#Now let us plot bar graph of item frequencies
import matplotlib.pyplot as plt
#here we are taking frequencies from zero to 11,you can try 0-15 or any other number
plt.bar(height=frequencies[0:11],x=list(range(0,11)))
plt.xticks(list(range(0,11)),items[0:11])
#plt.xticks,You can specify a rotation for the tick
#label in degrees or with keywords
plt.xlabel("items")
plt.ylabel("count")
plt.show()
import pandas as pd
#pip install mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
#Now let us try to establish association rule mining
#we have book list in the list format,we need to convert it in dataframe
#We designing rule on singke column childbks
book_series=pd.DataFrame(pd.Series(book['ChildBks']))
#book series has column having name 0,let us rename as transactions
book_series.columns=['Transactions']
#This is our input data to apply to apriori algorithm,it will generate 169
#is 0.0075(it must be between 0 to 1),you can give any number but must be between 0 and 1
x=book_series['Transactions']
frequent_itemsets=apriori(x,min_support=0.0075,max_len=4,use_colnames=True)
#you will get support values for 1,2,3 and 4 max items
#let us sort these support values
frequent_itemsets.sort_values('support',asccending=False,inplace=True)
#support values will be sorted in descending order
#Even EDA was also have the same trend,in EDA there was count
#and here it is support value
#we will generate association rules,This association
#rule will calculate all the matrix
#of each and every combination
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1)
#This generate association rules of size 1198X9 columns
# Comprises of antescends,consequences
rules.head(20)
rules.sort_values('lift',ascending=False).head(10)







