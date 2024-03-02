# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:09:25 2023

@author: Nilesh
"""

import pandas as pd
import numpy as np
from numpy import array
from scipy.linalg import svd
A=array([[1,0,0,0,2],[0,0,3,0,0],[0,0,0,0,0],[0,4,0,0,0]])
print(A)
#SVD
u,d,Vt=svd(A)
print(u)
print(d)
print(Vt)
print(np.diag(d))
#SVd applying to a dataset
import pandas as pd
data=pd.read_excel("c:/2-datasets/University_Clustering.xlsx")
data.head()
data=data.iloc[:,2:] #removes non numeric data
data
from sklearn.decomposition import TruncatedSVD
svd=TruncatedSVD(n_components=3)
svd.fit(data)
result=pd.DataFrame(svd.transform(data))
result.head()
result.columns='pc0','pc1','pc2'
result.head()

#scatter diagram
import matplotlib.pylab as plt
plt.scatter(x=result.pc0,y=result.pc1)
