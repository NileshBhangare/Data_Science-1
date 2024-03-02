# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:25:25 2023

@author: Nilesh
"""

sentence='we are learning TextMining from Sanjivani AI'
#if we want to know position of learning
sentence.index('learning')
#It will show learning is at postion 7
#This is going to show character position from 0 including

###################################################################
''' finding postion of word'''
#we want to know position of TextMining word
sentence.split().index('TextMining')
#It will split the words in list and count the position of perticular word
#if you want to see the list select sentence.split() and
#it will show at 3

############################################################
#suppose we want print any word in reverse order
sentence.split()[2][::-1]

sentence.split()[3][::-1]

#############################################################
#suppose want to print first and last word of the sentence
words=sentence.split()
first_word=words[0]
last_word=words[-1]
first_word
last_word

#################################################################
concat_word=first_word+" "+last_word
concat_word

##########################################################
#we want to print even words from sentence
words[::2]
even_words=[words[i] for i in range(len(words)) if i%2==0]
even_words

#want to display only AI
sentence
sentence[-3:]
#-3 will includes space and then AI

#reverse order sentence
sentence[::-1]

#Suppose we want to slect each word and print in reversed order
sentence.split()[::-1]
words
print(" ".join(i[::-1] for i in words))

###################################################################
''' ********************************* Tokenization ***************************'''
import nltk
nltk.download('punkt') #punkt is sentence tokenizer
from nltk import word_tokenize
words=word_tokenize("I am reading NLP Fundamentals")
print(words)

#######################################################
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

################################################################
import nltk
from nltk import word_tokenize
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
sentence4='We are learning NLP in python by SanjivaniAI based in India'
#first we will tokenize
words=word_tokenize(sentence4)
words=nltk.pos_tag(words)
i=nltk.ne_chunk(words,binary=True)
[a for a in i if len(a)==1]

########################################################
#sentence tokenization
from nltk.tokenize import sent_tokenize
sent=sent_tokenize('we are learning NLP in python. Delivered by SanjivaniAI. Do you know where it is located? It is in kopargaon')
sent

###########################################################
from nltk.wsd import lesk
nltk.download('wordnet')
nltk.download('omw-1.4')
sentence1='keep your savings in the bank '
print(lesk(word_tokenize(sentence1),'bank'))
#output-> Synset('savings_bank.n.02)
sentence2='It is so risky to drive over the banks of river'
print(lesk(word_tokenize(sentence2),'bank'))
#output-> Synset('bank.v.07')

##Synset('bank.v.07') a slope in the turn of a road or track.
#the outside is higher than the inside in order to reduce the
##
#'Bank' as multiple meaings. if you want to find exact meaning
#excute following code
#the definations for 'bank' can be seen here:
from nltk.corpus import wordnet as wn
for ss in wn.synsets('bank':print(ss,ss.defination()))
