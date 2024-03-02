# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 09:30:25 2023

@author: Nilesh
"""

from bs4 import BeautifulSoup as bs
import requests
link='https://www.amazon.in/UBON-BT-5100-Bluetooth-Lightweight-Sweat-Resistant/dp/B09QM1N978/ref=sr_1_2_sspa?crid=2PXALROJILRSJ&keywords=headphones+wireless&qid=1701749201&sprefix=%2Caps%2C234&sr=8-2-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1%27'
page=requests.get(link)
page
page.content
soup=bs(page.content,'html.parser')
print(soup.prettify())
title=soup.find_all('p',class_='_2-N8zT')
title
review_title=[]
for i in range(0,len(title)):
    review_title.append(title[i].get_text())
review_title
len(review_title)
for i in range(10):
    review_title.append('')
len(review_title)
#######################
rating=soup.find_all('div',class_='_3LWZlK _1BLPMq')
rating
rate=[]
for i in range(0,len(rating)):
    rate.append(rating[i].get_text())
rate
len(rate)    
for i in range(10):
    rate.append('')
len(rate)
###############################
review=soup.find_all('div',class_='t-ZTKy')
review
review_body=[]
for i in range(0,len(review)):
    review_body.append(review[i].get_text())
review_body
len(review_body)

##############################
import pandas as pd
from textblob import TextBlob
df=pd.DataFrame()
df['Review_title']=review_title
df['Rate']=rate
df['Review_body']=review_body
df.head

df.to_csv('c:/9-NLP/Flipkart_review.csv')

df=pd.read_csv('c:/9-NLP/Flipkart_review.csv')
#from sentiment import Polarity
text='It is very excellent garden'
blob=TextBlob(text)
pol=blob.sentiment.polarity
pol
df['Polarity']=df['Review_body'].apply(lambda x:TextBlob(x).sentiment.polarity )
df
