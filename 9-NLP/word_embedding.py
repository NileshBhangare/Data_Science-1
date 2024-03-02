# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 09:34:05 2023

@author: Nilesh
"""

#pip install gensim
#pip install python-Levenshtein
import gensim
import pandas as pd
df=pd.read_json('c:/2-datasets/Cell_Phones_and_Accessories_5.json',lines=True)
df
df.shape

#Simple Preprocessing and tokenization
review_text=df.reviewText.apply(gensim.utils.simple_preprocess)
review_text
#Lets us check first word of each review
review_text.loc[0]
#Let us check first row of dataframe
df.reviewText.loc[0]
#Trainning the Word2Vec Model
model=gensim.models.Word2Vec(
    window=10,
    min_count=2,
    workers=4
    )

'''
where window is how many words you are going to consider as sliding window
you can choose any count min_count-there must min 2 words in each sentence
workers:no.of threads

'''

#Bulid Vocabulary
model.build_vocab(review_text,progress_per=1000)
#Progress_per -> after 1000 words it shows progress
#Train the Word2Vec model
#It will take time
model.train(review_text,total_examples=model.corpus_count,epochs=model.epochs)
#save the model
model.save('c:/9-NLP/word2vec-amazon-cell-accessories-reviews-short.model')
#Finding similar words and similarity between words
model.wv.most_similar('bad')
model.wv.similarity(w1='cheap',w2='inexpensive')
model.wv.similarity(w1='good',w2='great')
model.wv.similarity(w1='good',w2='best')




