# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 09:12:25 2023

@author: Nilesh
"""

from sklearn.feature_extraction.text import TfidfVectorizer
corpus=['Thor eating pizza,Loki is eating pizza,Irinman ate pizza already',
        'Apple is announcing new iphone tomorrow',
        'Tesla is announcing new model-3 tomorrow',
        'Google is announcing new pixel-6 tomorrow',
        'Microsoft is announcing new surface tomorrow',
        'Amazon is announcing new eco-dot tomorrow',
        'I am eating biryani and you are eating grapes']
#let us create the vectorizer and fit the corpus and transform them accordingly
v=TfidfVectorizer()
v.fit(corpus)
transform_output=v.transform(corpus)
#lets print the vocabulary

print(v.vocabulary_)
#lets print the idf of each word

all_feature_names=v.get_feature_names_out()
for word in all_feature_names:
    #lets get the index in the vocabulary
    index=v.vocabulary_.get(word)
    #get the score
    idf_score=v.idf_[index]
    print(f"{word} : {idf_score}")
    
###################################################
import pandas as pd

df=pd.read_csv('c:/2-datasets/Ecommerce_data.csv')
print(df.shape)
df.head(5)

#check the distribution of labels
df['label'].value_counts()

#add the new column which gives a unique number to each of these label
df['label_num']=df['label'].map({'Household':0,'Books':1,'Electronics':2,
                                'Clothing & Accessories':3})

#Checking the results
df.head(5)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(
    df.Text,
    df.label_num,
    test_size=0.2,  # 20% of sample will go to test dataset
    random_state=2022, #2022 is one type of sampling technique,
                        #selects samples for testing randomly by tech 2022
    stratify=df.label_num  #stratify will convert imbalance data to 
    #balanced and then divided into multiclass variable
    )

print('Shape of X_train:',X_train.shape)
print('Shape og X_test:',X_test.shape)
y_train.value_counts()
y_test.value_counts()


###########################
#apply to the classifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

#create a pipeline object
clf=Pipeline([('vectorizer_tfidf',TfidfVectorizer()),
              ('KNN',KNeighborsClassifier())])

#fit with X_train and y_train
clf.fit(X_train,y_train)

#Get the predictions fpr x_test and store it in y_pred
y_pred=clf.predict(X_test)

#checking classification report
print(classification_report(y_test,y_pred))



