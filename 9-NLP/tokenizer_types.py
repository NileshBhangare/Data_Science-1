# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 09:02:17 2023

@author: Nilesh
"""

import re
sentence5='sharat twitted ,Wittnessing 70th republic day India from Rajpath, \new Delhi,Mesmorizing performance by Indian Army!'
re.sub(r'([^\s\w]|_)+',' ',sentence5).split()
#also called sentence cleaning
#extracting n-grams can be extracted using three techniques 
#1.custom defined function
#2. NLTK
#3.TextBlob

########################################################
#Extracting n-grams using custom defined function
import re
def n_gram_extrator(input_str,n):
    tokens=re.sub(r'([^\s\w]|_)+',' ',input_str).split()
    for i in range(len(tokens)-n+1):
        print(tokens[i:i+n])
    
n_gram_extrator('The cute little boy playing with kitten', 2)
n_gram_extrator('The cute little boy playing with kitten', 3)

#######################################################
''' 2.using NLTK '''
from nltk import ngrams
#extraction n-grams with nltk
list(ngrams('The cute little boy is playing with kitten'.split(),2))
list(ngrams('The cute little boy is playing with kitten'.split(),3))

###########################################################
''' 3.using TextBlob '''
from textblob import TextBlob
blob=TextBlob('The cute little boy is playing with kitten.')
blob.ngrams(n=2)
blob.ngrams(n=3)

############################################################
#Tokenization using keras
sentence5
from keras.preprocessing.text import text_to_word_sequence
text_to_word_sequence(sentence5)

########################################################
#Tokenization using TextBlob
from textblob import TextBlob
blob=TextBlob(sentence5)
blob.words
########################################################
#tweet Tokenizer
from  nltk.tokenize import TweetTokenizer
tweet_tokenizer=TweetTokenizer()
tweet_tokenizer.tokenize(sentence5)
########################################################
#Multiword tokenization
from nltk import MWETokenizer
sentence5
mwe_tokenizer=MWETokenizer([('republic','day')])
mwe_tokenizer.tokenize(sentence5.split())
mwe_tokenizer.tokenize(sentence5.replace("!",' ').split())
#############################################################
#Regular Expression Tokenizer
from nltk.tokenize import RegexpTokenizer
reg_tokenizer=RegexpTokenizer('\w+|\$[\d\.]+|\S+')
reg_tokenizer.tokenize(sentence5)

############################################################
#White space tokenizer
from nltk.tokenize import WhitespaceTokenizer
wh_tokenizer=WhitespaceTokenizer()
wh_tokenizer.tokenize(sentence5)

#############################################
#WordPunct Tokenization
from nltk.tokenize import WordPunctTokenizer
wp_tokenizer=WordPunctTokenizer()
wp_tokenizer.tokenize(sentence5)
##############################################
sentence6='I love playing cricket.Cricket Players practices hard in inning'
from nltk.stem import RegexpStemmer
regex_stemmer=RegexpStemmer('ing$')
' '.join(regex_stemmer.stem(wd) for wd in sentence6.split())
#############################################################
sentence7='Before eating ,it would be nice to sanitize your hands'
from nltk.stem.porter import PorterStemmer
ps_stemmer=PorterStemmer()
words=sentence7.split()
" ".join([ps_stemmer.stem(wd) for wd in words])
#######################################################
#lemmitization
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
nltk.download('wordnet')
lemmatizer=WordNetLemmatizer()
sentence8='The codes executed today are far better than what we execute generally'
words=word_tokenize(sentence8)
" ".join(lemmatizer.lemmatize(word) for word in words)

########################################################
#Sigularize and Pluralization
from textblob import TextBlob
sentence9=TextBlob('She shells seashells on the seashore')
words=sentence9.words
#we want to make word[2] i.e seashells in sigular form
sentence9.words[2].singularize()
#We want to word[5] in plural form
sentence9.words[5].pluralize()

#########################################################
#Language translation from spanish to english
from textblob import TextBlob
en_blob=TextBlob(u"muy bien")
en_blob.translate(from_lang='es',to='en') #es->spanish ,en->english
#Output -> very good

#######################################################
#Custome stopwords removal
sentence9='She shells seashells on the seashore'
from nltk import word_tokenize
custom_stop_words_list=['she','on','the','am','is']
words=word_tokenize(sentence9)
" ".join([word for word in words if word.lower() not in custom_stop_words_list])
#select the word that are not defined in list

######################################################
#extracting general features from raw text
#number of words
#detect presence of wh word
#polarity
#subjectivity
#langauge identification
########################################################
#To identify the number of words
import pandas as pd
df=pd.DataFrame([['The vaccine for covid-19 will be announced on 1 st August'],
                 ['Do you know how much expectations the world population is having from this research?'],
                 ['The risk of virus will be come to an end on 31st july']])
df.columns=['text']
df
#Now let us measure the number of words
from textblob import TextBlob
df['number_of_words']=df['text'].apply(lambda x:len(TextBlob(x).words))
df['number_of_words']

######################################################
#Detect presence of word wh
wh_words=set(['why','what','who','which','where','when','how'])
df['is_wh_words_present']=df['text'].apply(lambda x : True if len(set(TextBlob(str(x)).words).intersection(wh_words))>0 else False)
df['is_wh_words_present']

############################################################
#Polarity of the sentence
df['polarity']=df['text'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity']
sentence10='I like this example very much'
pol=TextBlob(sentence10).sentiment.polarity
pol
sentence10='This is fantastic example and I like it very much'
pol=TextBlob(sentence10).sentiment.polarity
pol
sentence10='This was helpful example but I would have prefer another one'
pol=TextBlob(sentence10).sentiment.polarity
pol
sentence10='This is my personal opinion that it was helpful example but I would prefer another one'
pol=TextBlob(sentence10).sentiment.polarity
pol
###########################################################
#Subjectivity-> sentence does not have context
df['subjectivity']=df['text'].apply(lambda x:TextBlob(str(x)).sentiment.subjectivity)
df['subjectivity']
##############################################################
#To find language of the sentence,this part of code will get
df['language']=df['text'].apply(lambda x:TextBlob(str(x)).detect_language())
df['language']=df['text'].apply(lambda x:TextBlob(str(x)).detect_language())

#####################################################
#NLP pipeline
# data aquisition ->text extraction and cleanup ->preprocessing ->feature engg


##########################################################
#Bag Of Words
#This BoW converts unstructured data to structered form
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
corpus=['at least seven indian pharma companies are working to develop vaccine against the corona virus.','The deadly virus that has already infected more than 14 million globally',
        'Bharat Biotech is the among the domastic pharma firm working on the corona virus vaccine in India']
bag_of_words_model=CountVectorizer()
print(bag_of_words_model.fit_transform(corpus).todense())
bag_of_words_df=pd.DataFrame(bag_of_words_model.fit_transform(corpus).todense())

#This will create dataframe
bag_of_words_df.columns=sorted(bag_of_words_model.vocabulary_)
bag_of_words_df.head()

#########################################################
bag_of_word_small=CountVectorizer(max_features=5)
bag_of_word_small_df=pd.DataFrame(bag_of_word_small.fit_transform(corpus).todense())
bag_of_word_small_df.columns=sorted(bag_of_word_small.vocabulary_)
bag_of_word_small_df.head()
