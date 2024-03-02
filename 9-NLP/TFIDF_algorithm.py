# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 08:50:32 2023

@author: Nilesh
"""

#TFIDF
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
corpus=['The mouse had a tiny little mouse',
        'The at saw the mouse','The cat catch the mouse',
        'The end of mouse story']
#Step 1 initialize the vector
cv=CountVectorizer()
#To count the total no of TF
word_count_vector=cv.fit_transform(corpus)
word_count_vector.shape
#Now next step is to apply IDF
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
#This matrix is in raw form,convert it inot dataframe
df_idf=pd.DataFrame(tfidf_transformer.idf_,index=cv.get_feature_names_out(),columns=['idf_weights'])
#Sort ascending
df_idf.sort_values(by=['idf_weights'])
