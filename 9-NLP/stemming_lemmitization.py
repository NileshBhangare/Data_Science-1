# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 19:21:48 2023

@author: Nilesh
"""
import nltk
#stemming
stemmer=nltk.stem.PorterStemmer()
stemmer.stem('Programming')
stemmer.stem('Programmed')
stemmer.stem('Jumping')
stemmer.stem('Jumped')
##################################################################
#lematizer
#lematizer looks into dict words
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
lemmatizer.lemmatize('Programmed')
lemmatizer.lemmatize('amazing')
lemmatizer.lemmatize('battling')
