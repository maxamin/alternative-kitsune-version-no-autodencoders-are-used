#!/usr/bin/env python
# coding: utf-8

# In[37]:


# Dataset cleaning done with IQR found on In [39]

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


# In[38]:


L_csv_path = os.path.join(PATH, L_FILE)
dataset_L = pd.read_csv(L_csv_path, dtype={"": int, "x": 'float64'})  

display(dataset_L.head())
display(dataset_L.info())
display(dataset_L.describe())


# In[39]:


import numpy as np
import matplotlib as plt
from scipy import stats

# tried modifying the quatiles to increase the range of data used... then tried to decrease the range of data used
# nothing helped for this

#quartile 1
quartile_one = dataset.quantile(0.05)

#quartile 3
quartile_three = dataset.quantile(0.95)

#range of data between quartiles 3 and 1
innerquartilerange = quartile_three - quartile_one

#modfiying the data based on the IQR for the original dataset
dataset_filt = dataset[~((dataset < (quartile_one - 1.5 * innerquartilerange)) |(dataset > (quartile_three + 1.5 * innerquartilerange))).any(axis=1)]
dataset_L_filt = dataset_L[~((dataset < (quartile_one - 1.5 * innerquartilerange)) |(dataset > (quartile_three + 1.5 * innerquartilerange))).any(axis=1)]

#z_score = np.abs(stats.zscore(dataset))
#dataset_filt = dataset[(z_score < 3).all(axis=1)]
#dataset_L_filt = dataset_L[(z_score < 3).all(axis=1)]
    
print("original dataset:")
display(dataset.info())
display(dataset_L.info())
print("new with IQR outlier filtering:")
display(dataset_filt.info())
display(dataset_L_filt.info())

print("checking it out")
display(dataset.head(5))
display(dataset_filt.head(5))


# In[40]:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error as mse

x_train, x_test, y_train, y_test = train_test_split(dataset_filt, dataset_L_filt.drop('Unnamed: 0', axis=1))


# In[41]:


logreg = LogisticRegression(multi_class='ovr', solver='lbfgs', n_jobs=11)
logreg.fit(x_train, y_train.values.ravel())

y_pred = logreg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))

y_tr_pred = logreg.predict(x_train)
print('Training Mean Absolute Error', mse(y_tr_pred,y_train))

print("Testing MSE = ", mse(y_pred,y_test))


# In[42]:


logreg_cv_score = cross_val_score(logreg, dataset_filt, dataset_L_filt.drop('Unnamed: 0', axis=1), cv=10, scoring='roc_auc', n_jobs=10)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAll AUC Scores:\n", logreg_cv_score)
print("Mean AUC Score - Logistic Regression:\n", logreg_cv_score.mean())


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
import statsmodels.api as sm

aa=0
ii=23

list_new_etc = []
feature_list_etc = []

for i in range(5):
    test = ExtraTreesClassifier(n_jobs=12)
    fit = test.fit(x_train.iloc[:,aa:ii], y_train.values.ravel())
    
    feat_importances = pd.Series(fit.feature_importances_, index=x_train.iloc[:,aa:ii].columns)
    
    print(feat_importances.nlargest(2))
    
    best = feat_importances.nlargest(2)
    
    list_new_etc=best.index.tolist()  

    feature_list_etc+=list_new_etc
    
    aa+=23
    ii+=23

print("\nFeature Selection done by ExtraTreesClassifier: ")
print(feature_list_etc)


# In[ ]:


x1_train, x1_test, y1_train, y1_test = train_test_split(dataset_filt.iloc[:,feature_list_etc], dataset_L_filt.drop('Unnamed: 0', axis=1))

logreg1 = LogisticRegression(multi_class='ovr', solver='lbfgs', n_jobs=11)
logreg1.fit(x1_train, y1_train.values.ravel())

y1_pred = logreg1.predict(x1_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg1.score(x1_test, y1_test)))

y1_tr_pred = logreg1.predict(x1_train)
print('Training Mean Absolute Error', mse(y1_tr_pred,y1_train))

print("Testing MSE = ", mse(y1_pred,y1_test))


# In[ ]:


logreg1_cv_score = cross_val_score(logreg1, dataset_filt.iloc[:,feature_list_etc],dataset_L_filt.drop('Unnamed: 0', axis=1), cv=10, scoring='roc_auc', n_jobs=11)

print("Confusion Matrix:")
print(confusion_matrix(y1_test, y1_pred))
print("\nClassification Report:")
print(classification_report(y1_test, y1_pred))
print("\nAll AUC Scores:\n", logreg1_cv_score)
print("Mean AUC Score - Logistic Regression:\n", logreg1_cv_score.mean())


# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from numpy import set_printoptions
import heapq

aa=0
ii=23

list_new_skb = []
feature_list_skb = []

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
    
    aa+=23
    ii+=23

print("\nFeature Selection done by SelectKBest: ")
print(feature_list_skb)


# In[ ]:


x2_train, x2_test, y2_train, y2_test = train_test_split(dataset_filt.iloc[:,feature_list_skb], dataset_L_filt.drop('Unnamed: 0', axis=1))

logreg2 = LogisticRegression(multi_class='ovr', solver='lbfgs', n_jobs=11)
logreg2.fit(x2_train, y2_train.values.ravel())

y2_pred = logreg2.predict(x2_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg2.score(x2_test, y2_test)))

y2_tr_pred = logreg1.predict(x2_train)
print('Training Mean Absolute Error', mse(y2_tr_pred,y2_train))

print("Testing MSE = ", mse(y2_pred,y2_test))


# In[ ]:


logreg2_cv_score = cross_val_score(logreg2, dataset_filt.iloc[:,feature_list_skb],dataset_L_filt.drop('Unnamed: 0', axis=1), cv=10, scoring='roc_auc', n_jobs=11)

print("Confusion Matrix:")
print(confusion_matrix(y2_test, y2_pred))
print("\nClassification Report:")
print(classification_report(y2_test, y2_pred))
print("\nAll AUC Scores:\n", logreg2_cv_score)
print("Mean AUC Score - Logistic Regression:\n", logreg2_cv_score.mean())


# In[ ]:




