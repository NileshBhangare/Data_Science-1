# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 20:28:45 2023

@author: Nilesh
"""

''' 
Created By :Nilesh


********************* Recommendation System on Entertainment.csv *************
Problem Statement :
    The Entertainment Company, which is an online movie watching platform,
    wants to improve its collection of movies and showcase those that are 
    highly rated and recommend those movies to its customer by their movie
    watching footprint. For this, the company has collected the data and 
    shared it with you to provide some analytical insights and also to 
    come up with a recommendation algorithm so that it can automate its 
    process for effective recommendations. 
    The ratings are between -9 and +9.
    
Business Objective:
    Maximize : Increase User Satisfaction.
               Increase accuracy while recommending movies to user as per thier likes and preferences.
    Minimize: Minimizing rate of recommending wrong movies
    
    Business Constraints:
        The recommended movies should be available currently.
        
        
*********************** Data Dictionary **************************
Name of feature       Description          Type                 Relevance

Id                    Id of Movie        Discrete Data          Irrelevant
Titles                Title of Movie      Nominal data          Relevant
category             Category of movie   Nominal Data           Relevant
Reviews               Rating of movie    Continuous data        Relevant


'''
import pandas as pd
en=pd.read_csv('c:/2-datasets/Entertainment.csv')
en
en.shape
#The dataset contain 51 rows and 4 columns

en.columns
#It have 4 columns
#1.Id
#2.Titles
#3.category
#4.Reviews

en.dtypes
'''Id            Discrete Data(int)
Titles        Nominal data(object/string)
Category      Nominal data(object/string)
Reviews       continuous Data(float)'''

'''*************************EDA*****************************'''
en.isnull().sum()
#It does not contain any null values

en.duplicated().sum()
#It does not contain duplicate value

#Dropping Id Column as it is Irrelevant
en.drop(['Id'],axis=1,inplace=True) 

en.describe()
#Describe method will give you the 5 number summery of Review columns as 
#it is only the have the continuous or numerical type of data

#It will give you that the data is variated between -9 and 99 
#From this we can say that data is rightskewed 
#with mean 36.28 and std 49.03


#Cheking for outlier
import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(en['Reviews'])
#There is no outlier


'''**************************** Recommendation System **********************'''
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(stop_words='english')
#It is going to create TfidfVectorizer to seperate all stop words.
#It is going to seperate
#out all words from the row


#now let us check is there any null value
en['Reviews'].isnull().sum()
#there are 0 null values

#suppose one movie has got genre Drama,Romance...
#now let us create tfidf_matrix
tfidf_matrix=tfidf.fit_transform(en['Reviews'].astype(str))
tfidf_matrix.shape
#You will get 51,25
#It has created sparse matrix,it means
#that we have 25 genre
#on this particular matrix,

#we want to do item based recommendation,if a user has

from sklearn.metrics.pairwise import linear_kernel#This is for measuring similarity
cosine_sim_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)
#each element of tfidf_matrix is compared
#with each element tfidf_matrix only
#output will be similarity matric of size 51 X 51 size
#here in cosine_sim_matrix
#there are no movie names only index are provided
#we will try to map movie name with movie index given
#for that purpose custom function is written
en_index=pd.Series(en.index,index=en['Titles']).drop_duplicates()
#we are converting anime_index into series format,we want index and corresponding names
entertainment_id=en_index['Heat (1995)']
entertainment_id
def get_recommendations(Name,topN):
    
    entertainment_id=en_index[Name]
    #for that purpose we are applying cosine_sim_matrix to enumerate function
    #Enumerate function create a object,
    #which we need to create in list form
    #we are using enumerate function,
    #what enumerate does,suppose we have given
    cosine_scores=list(enumerate(cosine_sim_matrix[entertainment_id]))
    #the cosine scores captured,we want to arrange in descending order so that
    #we can recomment top 10 based on highest similarity i.e score
    #if we will check the cosine score, it comprises of index:cosine score
    #x[0]=index and x[1] is cosine score
    #we want arrange tuples according to decreasing order
    #of the score not index
    #sorting the cosine_similarity scores based on scores i.e x[1]
    cosine_scores=sorted(cosine_scores,key=lambda x:x[1],reverse=True)
    #get the scores of top N most similar movies
    #To capture topN movies,you need to give topN+1
    cosine_scores_N=cosine_scores[0:topN+1]
    #getting the movie index
    en_idx=[i[0] for i in cosine_scores_N]
    #getting cosine score
    en_scores=[i[1] for i in cosine_scores_N]
    #we are going to use this information to create a dataframe
    #create a empty dataframe
    en_similar_show=pd.DataFrame(columns=['Title'])
    en_similar_show['Title']=en.loc[en_idx,'Titles']
    #assign score to score column
    #while assigning values,it is by default capturing original index of the
    #we want to reset the index
    en_similar_show.reset_index(inplace=True)
    print(en_similar_show)

get_recommendations('Heat (1995)', topN=10)




