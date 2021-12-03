#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Part e) Feature Extraction

import os
import tarfile
import urllib

import pandas as pd

PATH = "arp_mitm"

FILE = "ARP MitM_dataset-002.csv"
L_FILE = "ARP MitM_labels.csv"

csv_path = os.path.join(PATH, FILE)
dataset = pd.read_csv(csv_path, header=None)  

display(dataset.head())
display(dataset.info())
display(dataset.describe())

L_csv_path = os.path.join(PATH, L_FILE)
dataset_L = pd.read_csv(L_csv_path, dtype={"": int, "x": 'float64'})  

display(dataset_L.head())
display(dataset_L.info())
display(dataset_L.describe())


# In[3]:


from sklearn.model_selection import train_test_split

# Splitting the dataset to training and testing data
x_train, x_test, y_train, y_test = train_test_split(dataset, dataset_L.drop('Unnamed: 0', axis=1))


# In[7]:


# Feature Selection done with Extra Trees Classifier

from sklearn.ensemble import ExtraTreesClassifier
import statsmodels.api as sm

aa=0
ii=23

list_new_etc = []
feature_list_etc = []
histogram_list = []

for i in range(5):
    test = ExtraTreesClassifier(n_jobs=12)
    fit = test.fit(x_train.iloc[:,aa:ii], y_train.values.ravel())
    
    feat_importances = pd.Series(fit.feature_importances_, index=x_train.iloc[:,aa:ii].columns)
    
    print(feat_importances.nlargest(2))
    
    best = feat_importances.nlargest(2)
    
    list_new_etc=best.index.tolist()  

    feature_list_etc+=list_new_etc
    
    histogram_list += feat_importances.values.tolist()
    
    aa+=23
    ii+=23

print("\nFeature Selection done by ExtraTreesClassifier: ")
print(feature_list_etc)


# In[11]:


# Plots showing the top 2 features selected from each of the 5 timestep using Extra Trees Classifier

import matplotlib.pyplot as plt 

aa=0
ii=23

for i in range(5):
    #histogram_list = feat_importances[aa:ii].values.tolist()
    for bb in range(aa,ii):
        plt.bar(bb, histogram_list[bb])#, label='Features')
        #plt.hist(best.values, label='2 most important')
        plt.xlabel("Feature {}".format(i), size=14)
        plt.ylabel("Magnitude of Feature", size=14)
        #plt.legend(loc="upper left")
        plt.title("ExtraTreesClassifier(): Timestep {} Features".format(i))
    plt.show()
    
    aa+=23
    ii+=23


# In[12]:


# Feature Selection done with Select K Best Classifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from numpy import set_printoptions
import heapq

aa=0
ii=23

list_new_skb = []
feature_list_skb = []
histogram_list2 = []

for i in range(5):
    test = SelectKBest(score_func=f_classif, k=2)
    fit = test.fit(x_train.iloc[:,aa:ii], y_train.values.ravel())
    scores = pd.DataFrame(fit.scores_)
    index = pd.DataFrame(x_test.columns)
    
    featureScores = pd.concat([index,scores],axis=1)
    featureScores.columns = ['Specs','Score'] 
    print(featureScores.nlargest(2,'Score'))  
    
    best=featureScores.nlargest(2,'Score')
    
    list_new_skb=best['Specs'].values.tolist()
    list_new_skb[0]+=aa
    list_new_skb[1]+=aa    

    feature_list_skb+=list_new_skb
    
    ofc = featureScores.dropna()
    #print(ofc)
    histogram_list2 += ofc['Score'].values.tolist()
    #print(histogram_list2)
    
    aa+=23
    ii+=23

print("\nFeature Selection done by SelectKBest: ")
print(feature_list_skb)


# In[13]:


# Plots showing the top 2 features selected from each of the 5 timestep using Select K Best Classifier

aa=0
ii=23

#print(histogram_list2)

for i in range(5):
    #histogram_list = feat_importances[aa:ii].values.tolist()
    for bb in range(aa,ii):
        plt.bar(bb, histogram_list2[bb])#, label='Features')
        #plt.hist(best.values, label='2 most important')
        plt.xlabel("Feature {}".format(i), size=14)
        plt.ylabel("Magnitude of Feature", size=14)
        #plt.legend(loc="upper left")
        plt.title("SelectKBest(): Timestep {} Features".format(i))
    plt.show()
    
    aa+=23
    ii+=23


# In[ ]:




