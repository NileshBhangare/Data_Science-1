# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:04:31 2023

@author: Vaibhav Bhorkade 

Dataset s
"""

import pandas as pd
import numpy as np

df=pd.read_csv('D:/2-datasets/loan.csv')
print(df)
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
df['id'].astype('float')
df.dtypes

df2=df.astype({'member_id':'float','loan_amnt':'float'})
df2.dtypes
# list col
cols=['id','member_id']
df[cols]=df[cols].astype('int')
df.dtypes
# Errors ignore
df=df.astype({'id':str},errors='ignore')
df
# errors raise
df=df.astype({'id':str},errors='raise')
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
df['int_rate']
df['int_rate'][4]

# Accessing two columns
df[['loan_amnt','funded_amnt']]
df[['loan_amnt','term','grade']]

# accessing rows and columns-all
df[:]
df[1:1]
df[:2]
df[0:1]

# rename()
df=pd.read_csv('D:/2-datasets/loan.csv')
print(df)
df.columns

df2=df.rename({'id':'ID'},axis=1)
df2
df2=df.rename({'id':'ID'},axis='columns')
df2
df2=df.rename(columns={'id':'ID','member_id':"MID"})
df2
####################################################################################
import pandas as pd
import numpy as np

df=pd.read_csv('D:/2-datasets/bank_data.csv')
print(df)
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
df['age'].astype('float')
df.dtypes

df2=df.astype({'age':'float','loan':'float'})
df2.dtypes
# list col
cols=['age','loan']
df[cols]=df[cols].astype('int')
df.dtypes
# Errors ignore
df=df.astype({'age':str},errors='ignore')
df
# errors raise
df=df.astype({'married':int},errors='raise')
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
df['duration']
df['duration'][4]

# Accessing two columns
df[['age','duration']]
df[['age','duraton','balance']]

# accessing rows and columns-all
df[:]
df[1:1]
df[:2]
df[0:1]

# rename()
df=pd.read_csv('D:/2-datasets/bank_data.csv')
print(df)
df.columns

df2=df.rename({'age':'A'},axis=1)
df2
df2=df.rename({'age':'AGE'},axis='columns')
df2
df2=df.rename(columns={'age':'AGE','balance':"BALANCE"})
df2

####################################################################################
import pandas as pd
import numpy as np

df=pd.read_csv('D:/2-datasets/crime_data.csv')
print(df)
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
df['Assault'].astype('float')
df.dtypes
df2=df.astype({'Murder':'float','Assault':'float'})
df2.dtypes
# list col
cols=['Murder','Assault']
df[cols]=df[cols].astype('int')
df.dtypes
# Errors ignore
df=df.astype({'Murder':str},errors='ignore')
df
# errors raise
df=df.astype({'Murder':int},errors='raise')
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
df['Murder']
df['Murder'][4]

# Accessing two columns
df[['Murder','Assault']]

# accessing rows and columns-all
df[:]
df[1:1]
df[:2]
df[0:1]

# rename()
df=pd.read_csv('D:/2-datasets/crime_data.csv')
print(df)
df.columns

df2=df.rename({'Murder':'A'},axis=1)
df2
df2=df.rename({'Murder':'MA'},axis='columns')
df2
df2=df.rename(columns={'Murder':'A','UrbanPop':"U"})
df2

