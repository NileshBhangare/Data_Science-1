# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:06:52 2023

@author: Nilesh


Created By : Nilesh
"""
''' 
----------------------------- Recommendation Engine for game.csv ------------------------

Problem Statement : Build a recommender system with the given data .

This dataset is related to the video gaming industry and a survey was conducted to build a 
recommendation engine so that the store can improve the sales of its gaming DVDs. 
Build a Recommendation Engine and suggest top selling DVDs to the store customers.
   
Business Objective: 
    Maximize: Increase the sales of gaming DVDs
    Minimize : Error while recommending Gaming DVDs
    
    Business Constraints: Accuracy of prediction and recommendation


*************************Data Dictionary************************

Name of feature           Description              Type             Relevance

userId                    Id of user               Discrete Data     Irrelevant
Game                      Name of Game            Nominal Data        Relevant
Ratings                   Rating of game          Continuous          Relevant


'''
import pandas as pd

game=pd.read_csv('c:/2-datasets/game.csv')
game
game.columns
#it contains 3 columns
#1.UserId
#2.Game
#3.ratings

game.dtypes
#userId -> Discrete Data (int)
#Game -> Nominal Data(object/string)
#Ratings -> Continuous Data(float)

game.shape
#it contains 5000 rows and 3 columns

'''***************************EDA****************************'''
game.describe()
#As in our dataset we have 2 columns with quantitative data and 1 coulmn with
#Qualitative Data
#As describe method is only works on quantitative data 
#so it will result the 5 number summery of only 2 columns userid and rating

#As userId is Irrelevant so we need to drop it
#Droping the column userId
game.drop(['userId'],axis=1,inplace=True)


#Now again performing describe on dataframe
game.describe()
#describe is giving 5 number summery of ratings column
#From 5 number summery of ratings ,we got the data is variated from 0.5 to 5
#and having mean 3.59 ,std is 0.99

#Checking for null values
game.isnull().sum()
#It does not contain any null value

#Checking for duplicate values
game.duplicated().sum()
#It contains 429 duplicates


#Dropping of duplicates
game.drop_duplicates(inplace=True)

#Now again checking for duplicates
game.duplicated().sum()
#All duplicates are removed

game.shape
#Now dataframe contain 4571 rows and 2 columns

################################################

#Converting data of game column into discrete data
#using one hot encoding

game_new=pd.get_dummies(game)
game_new

game_new.shape
#after creating dummy variables,we have 4571 rows and 3439 columns

game_new.columns

#Droping of one column as converting dummy variable of n columns 
#we need to drop one column to get n-1 columns 

game_new.drop(game_new['game_flower'],axis=1,inplace=True)

#checking 5 number summery
des_game=game_new.describe()
#From describe,we get that data is not normalized as mean of rating is 3.57 and
#other columns have mean almost 1
#So we need to normalized data

def norm_fun(i):
    x=(i-i.min()/i.max()-i.min())
    return x

game_norm=norm_fun(game_new)
game_norm.describe()


'''****************************Recommendation System******************'''
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(stop_words='english')
#Creating ifidf matrix
tfidf_matrix=tfidf.fit_transform(game['rating'].astype(str))
tfidf_matrix.shape
#Tfidf matrix contains 4571 rows and 3 columns
#We have created sparse matrix containg three different rating value

#Now we are applying item recommendation
from sklearn.metrics.pairwise import linear_kernel
cosine_sim_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)
#It will compared tfidf matric with each value of tfidf matrix value
#It will create matrix of size 4571 X 4571

game_index=pd.Series(game.index,index=game['game']).drop_duplicates()

game_id=game_index['SoulCalibur']
game_id

def get_game_recommendations_with_ratings(name, topN):
    
    # sort the game based on rating of game
    sorted_games = game.sort_values(by='rating', ascending=False)
    #Top N games with highest ratings
    top_game_recommendations = sorted_games.head(topN+1)
    
    #It is taking by original index while assigning so
    #Changing the index
    top_game_recommendations.reset_index(drop=True, inplace=True)
    
    print(top_game_recommendations)

#enter your anime and number of animes to be recommended
get_game_recommendations_with_ratings('NASCAR Heat', topN=5)





    