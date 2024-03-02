# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 20:09:31 2023

@author: Nilesh
"""

'''
Created By: Vaishnavi Pawar

-------------------------- PCA Assignement on Wine.csv ----------------------------

Problem Statement : Perform hierarchical and K-means clustering on the dataset. 
                    After that, perform PCA on the dataset and extract the 
                    first 3 principal components and make a new dataset 
                    with these 3 principal components as the columns. 
                    Now, on this new dataset, perform hierarchical and 
                    K-means clustering. Compare the results of clustering 
                    on the original dataset and clustering on the principal 
                    components dataset (use the scree plot technique to 
                    obtain the optimum number of clusters in K-means.
                    
Business Objective: 
    Maximize: 
    Minimize:
        
    Business Constraints:
        
        
Data Dictionary :
    
Name Of feature           Description               Type               Relevance

Type                      
Alcohol
Malic
Ash
Alcalinity
Magnesium
Phenols
Flavanoids
Nonflavanoids
Proanthocyanins
Color
Hue
Dilution
Proline
 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
wine=pd.read_csv('c:/2-datasets/wine.csv')
wine
wine.shape
wine.columns
wine.dtypes
