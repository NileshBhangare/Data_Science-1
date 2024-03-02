# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 19:18:19 2023

@author: Nilesh
"""

'''
Created By: Vaishnavi Pawar

1.Using tokenization , Extract all money transaction from below 
sentence along with currency. Output should be,
    wo $
    500 €
'''
import re
transactions='Tony gave two $ to Peter, Bruce gave 500 € to Steve'
re.findall(r'[a-z]{2} \$ |\d+ \€',transactions)

#here findall will find the string that matches the regular expression

'''
Output:
    ['wo $ ', '500 €']
    
'''

'''
2.Use stemming for following docs
doc = nlp("Mando talked for 3 hours although talking isn't his thing")
doc = nlp("eating eats eat ate adjustable rafting ability meeting better")
'''

import nltk
from nltk.tokenize import word_tokenize
doc1="Mando talked for 3 hours although talking isn't his thing"
doc2="eating eats eat ate adjustable rafting ability meeting better"

#On doc1
words=word_tokenize(doc1)
stemmer=nltk.stem.PorterStemmer()
stemming_doc1=[stemmer.stem(word) for word in words]
final_stemming=' '.join(stemming_doc1)
final_stemming

'''output :
    mando talk for 3 hour although talk is n't hi thing
    
'''

#on doc2

words1=word_tokenize(doc2)
stemming_doc2=[stemmer.stem(word) for word in words1]
final_stemming_doc2=' '.join(stemming_doc2)
final_stemming_doc2

''' Output :
        eat eat eat ate adjust raft abil meet better
'''

'''
3.convert these list of words into base form using Stemming and Lemmatization 
    and observe the transformations.
    #using stemming in nltk
    lst_words = ['running', 'painting', 'walking', 'dressing', 'likely',
                 'children', 'whom', 'good', 'ate', 'fishing']
    #using lemmatization in spacy
    doc = nlp("running painting walking dressing likely children 
              who good ate fishing")

'''

import nltk
from nltk.tokenize import word_tokenize
lst_words=['running','painting','walking','dressing','likely','children',
           'whom','good','ate','fishing']

stemmer=nltk.stem.PorterStemmer()
stemmed_words=[stemmer.stem(word) for word in lst_words]
stemmed_words

'''
Output :
        ['run','paint','walk','dress','like','children','whom','good',
         'ate','fish']
In this, all words with ing and ly are converted into its base form ,
but ate is not converted into its base form eat.        
        
'''
#lemmitization
nltk.download("wordnet")
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmitizer=WordNetLemmatizer()
doc="running painting walking dressing likely children who good ate fishing"
lemma_words=word_tokenize(doc)
lemma_words
lemmitized_words=[lemmitizer.lemmatize(word) for word in lemma_words]
lemmitized_words
final_lemma_words=' '.join(lemmitized_words)
final_lemma_words

'''
Output:
    ['running','painting','walking','dressing','likely','child','who',
     'good','ate','fishing']
    
In lemmatization, as running and other words with ing and  ly are already in
its base form thats why it remain as it is. only children is change to child.
'''

'''
4.convert the given text into it's base form using both stemming and 
lemmatization
text = """Latha is very multi talented girl.She is good at many skills like dancing, running, singing, playing.She also likes eating Pav Bhagi. she has a 
habit of fishing and swimming too.Besides all this, she is a wonderful at cooking too.
"""
'''
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
text = """Latha is very multi talented girl.She is good at many skills 
like dancing, running, singing, playing .She also likes eating Pav Bhagi.
she has a habit of fishing and swimming too.Besides all this, 
she is a wonderful at cooking too.
"""
#Stemming
words=word_tokenize(text)
stemmer=nltk.stem.PorterStemmer()
stemming_words=[stemmer.stem(word) for word in words]
stemming_sentence=' '.join(stemming_words)
stemming_sentence

#After stemming result is 
'''
'latha is veri multi talent girl.sh is good at mani skill like danc , 
run , sing , play .she also like eat pav bhagi . 
she ha a habit of fish and swim too.besid all thi , 
she is a wonder at cook too .'
'''
#lemmatization
lemmatizer=WordNetLemmatizer()
lemmatization_words=[lemmatizer.lemmatize(word) for word in words]
lemmatized_sentence=' '.join(lemmatization_words)
lemmatized_sentence

#After lemmatization , the result is
'''
'Latha is very multi talented girl.She is good at many skill like 
dancing , running , singing , playing .She also like eating Pav Bhagi .
she ha a habit of fishing and swimming too.Besides all this , 
she is a wonderful at cooking too .'
'''
#########################################################

'''
5. You are parsing a news story from cnbc.com. News story is stores in 
    news_story.txt which is on whatsapp. You need to, 
        1.	Extract all NOUN tokens from this story. You will have to read the file in python first to collect all the text and then extract NOUNs in a python list
        2.	Extract all numbers (NUM POS type) in a python list
        3.	Print a count of all POS tags in this story

'''
with open('news_story.txt') as file:
    sentence=file.read()
    
sentence           
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

#1.
tokenize_words=word_tokenize(sentence)
pos_words=pos_tag(tokenize_words)

nouns = [word for word ,pos in pos_words if pos.startswith('N')]
nouns

#It is displaying all nouns from the file


#2.
tokenize_words=word_tokenize(sentence)
pos_words=pos_tag(tokenize_words)
number=[word for word,pos in pos_words if pos=='CD']
number
#It is displaying all numbers are there in file

#3
from collections import Counter
tokenize_words=word_tokenize(sentence)
pos_words=pos_tag(tokenize_words)
count=Counter(word for word,pos in pos_words) 
count
#it is displaying count of each word present in file



