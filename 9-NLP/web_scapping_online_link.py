# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 09:46:10 2023

@author:Nilesh
"""

from bs4 import BeautifulSoup as bs
import requests
link='https://sanjivanicoe.org.in/index.php/contact'
page=requests.get(link)
page
#<Response [200]> it means connection is successfully established
page.content
#You will get all html source code but very crowdy text
#let us apply html parser
soup=bs(page.content,'html.parser')
soup
#Now the text is clean but not upto the expectations
#Now let us apply prettify methos
print(soup.prettify())
#The text is neat and clean
list(soup.children)
#Finding all contents using tab
soup.find_all('p')
#suppose you want to extract contents from
#1st row
soup.find_all('p')[1].get_text()
#Contents from second row
soup.find_all('p')[2].get_text()
#finding text using class
soup.find_all('div',class_='table')
