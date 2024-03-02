# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 09:19:51 2023

@author: Nilesh
"""

from bs4 import BeautifulSoup
soup=BeautifulSoup(open('c:/2-datasets/sample_doc.html'),'html.parser')
print(soup)
#it is going to show all the html contents extracted
soup.text
#it will show only text
soup.contents
#it is going to show all the html contents extracted
soup.find('address')
soup.find_all('address')
soup.find_all('q')
soup.find_all('b')
table=soup.find('table')
table

for row in table.find_all('tr'):
    columns=row.find_all('td')
    print(columns)
    
#it will show all the rows except 1st row as 1st row is in 'th' tag

table.find_all('tr')[3].find_all('td')[2]

    