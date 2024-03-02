# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:35:51 2023

@author: Vaibhav Bhorkade
"""

import pandas as pd

df=pd.read_csv("D:/2-datasets/Seeds_data.csv")
print(df)

df.columns
# Pandas.DataFrame.query() 
# query all rows with Courses equals 'spark'

df2=df.query("Area==13.84")
df2

# not equal to 


df2=df.query("Area!=13.84")
df2


# Derive New column from Existing columns
df=pd.read_csv("D:/2-datasets/Seeds_data.csv")

df2=df.assign(Discount_Percentage=lambda x:x.length + x.Width / 100)
df2

# Get the number of rows in DataFrame
rows_count=len(df.index)
rows_count

rows_count=len(df.axes[0])
rows_count

column_count=len(df.axes[1])
column_count

# another method to find rows and col
rows_count=df.shape[0]
rows_count

column_count=df.shape[1]
column_count

# data types
df.dtypes
# rows and columns
df.shape
# rows*columns
df.size
# columns
df.columns
df.columns.values
# describe
df.describe()
# converting all types to possible types
df=df.convert_dtypes()
df.dtypes
# Converting all into same types
df=df.astype(str)
df.dtypes

# Change type of one columns
df['Area'].astype('float')
df.dtypes

df2=df.astype({'Area':'float','length':'float'})
df2.dtypes
# list col
cols=['Area','length']
df[cols]=df[cols].astype('int')
df.dtypes
# Errors ignore
df=df.astype({'Area':str},errors='ignore')
df
# errors raise
df=df.astype({'Area':str},errors='raise')
df

# creating data frame to csv
df.to_csv("Temp.csv")
df=pd.read_csv("Temp.csv")
df
# index
df.index
# describe
df.describe()
df.dtypes
df.columns

# Accessing one column 
df['length']
df['length'][4]

# Accessing two columns
df[['Area','length']]
df[['Area','Perimeter','length']]

# accessing rows and columns-all
df[:]
df[1:1]
df[:2]
df[0:1]

# Droping columns and rows
# Drop rows by labels
df1=df.drop([1,2])
df1

# Drop rows by index/position
df1=df.drop(df.index[1])
df1

df1=df.drop(df.index[[1,2]])
df1

# Delete rows by index range
df1=df.drop(df.index[2:])
df1


# only 0 th rows deleted
df1=df.drop(0)
df1
# delete rows 0 and 3
df1=df.drop([0,3])
df1
# delete 0,1 ,2 
df1=df.drop(range(0,3))
print(df1)

''' Droping of column '''
df=pd.DataFrame(technologies)
print(df)

# Drop columns by name
df1=df.drop(['Fee'],axis=1)
print(df1)
df1=df.drop(['Duration'],axis=1)
print(df1)

#'Labels' for drop the column
df1=df.drop(labels=['Fee'],axis=1)
print(df1)

# columns
df1=df.drop(columns=['Fee'],axis=1)
print(df1)

# Drop column using index
df1=df.drop(df.columns[1],axis=1)
print(df1)

# Original dataframe to change then using working -> inplace=True
df=df.drop(df.columns[2],axis=1,inplace=True)
print(df)

df2=df.drop(['Area','length'],axis=1)
print(df2)


# Drop two or more column by index

df2=df.drop(df.columns[[0,1]],axis=1)
print(df2)

# Drop columns from list 
liscol=["Area","length"]
df2=df.drop(liscol,axis=1)
df2

'''
df=df.drop(liscol,axis=1,inplace=True)
print(df)
'''

# iloc-> [startrow:lastrow,startcol:endcol]
df2=df.iloc[:,0:2]
print(df2)

# iloc and loc are slicing operator
# first slice[:] indicate to return all rows
# second slice [:] about columns

# for 0 and 1 rows and all coluns
df2=df.iloc[0:2,:] 
df2

# slicing Specific rows and specifc columns
# 1 row and 1,2 columns
df3=df.iloc[1:2,1:3]
df3

# only select rows->one series only
df2=df.iloc[2]
df2

 
df2=df.iloc[[2,3,5]] # select rows by index list
df2
df2=df.iloc[1:5] # Select rows by integer index list
df2
df2=df.iloc[:1] # select first row
df2
df2=df.iloc[:3] # select first 3 rows
df2
df2=df.iloc[-1:] # Select last row
df2
df2=df.iloc[-3:] # select last 3 rows
df2
df2=df.iloc[::2] # select alternate rows
df2


'''
Using loc to take column slices
df.loc[:,start:stop:step]
'''


# Select multiple col'
df2=df.loc[:,['Area','length','Width']]
df2
# Select only two column
df2=df.loc[:,'Area':'length']
df2
# Select column by range
df2=df.loc[:,'Area':]
df2
# Select col upto 'lengtn'xyuuu
df2=df.loc[:,:'length']
df2