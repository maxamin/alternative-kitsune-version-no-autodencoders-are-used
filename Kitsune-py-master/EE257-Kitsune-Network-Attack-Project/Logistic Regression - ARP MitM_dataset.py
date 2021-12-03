#!/usr/bin/env python
# coding: utf-8

# In[86]:


# Part f) Model Development

import os
import tarfile
import urllib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error as mse

# gathering the dataset

PATH = "arp_mitm"

FILE = "ARP MitM_dataset-002.csv"
L_FILE = "ARP MitM_labels.csv"

csv_path = os.path.join(PATH, FILE)
dataset = pd.read_csv(csv_path, header=None)  

display(dataset.head())
display(dataset.info())
display(dataset.describe())


# In[48]:


# gathering classified output data for dataset

L_csv_path = os.path.join(PATH, L_FILE)
dataset_L = pd.read_csv(L_csv_path, dtype={"": int, "x": 'float64'})  

display(dataset_L.head())
display(dataset_L.info())
display(dataset_L.describe())


# In[49]:


# NO FEATURE SELECTION
# gathering testing and training data
x_train, x_test, y_train, y_test = train_test_split(dataset, dataset_L.drop('Unnamed: 0', axis=1))


# In[50]:


# NO FEATURE SELECTION
# running logistic regression and training the model using fit with the training data
logreg = LogisticRegression(multi_class='auto', solver='lbfgs', n_jobs=11)
logreg.fit(x_train, y_train.values.ravel())


# In[68]:


# NO FEATURE SELECTION
# predicting the training and test data
y_tr_pred = logreg.predict(x_train)
y_pred = logreg.predict(x_test)


# In[95]:


# NO FEATURE SELECTION
# finding training and testing mse for the predictions found in the previous lines of code
print("Training MSE = ", mse(y_tr_pred,y_train))
print("Testing MSE = ", mse(y_pred,y_test))


# In[69]:


# NO FEATURE SELECTION
# finding the classification reports and accuracy scores for the training and testing datasets
print("\nTRAINING DATA:")
print(classification_report(y_train, y_tr_pred, target_names=['not malicous', 'malicous']))
print(metrics.accuracy_score(y_train, y_tr_pred))

print("\nTEST DATA:")
print(classification_report(y_test, y_pred, target_names=['not malicous', 'malicous']))
print(metrics.accuracy_score(y_test, y_pred))


# In[70]:


# NO FEATURE SELECTION
# confusion matrices for training and testing set
print("Number of 0's and 1's in y_train dataset:")
print(y_train['x'].value_counts())
print("\nCONFUSION MATRIX FOR TRAINING SET: \n {}".format(confusion_matrix(y_train, y_tr_pred)))

print("\nNumber of 0's and 1's in y_test dataset:")
print(y_test['x'].value_counts())
print("\nCONFUSION MATRIX FOR TESTING SET: \n {}".format(confusion_matrix(y_test, y_pred)))


# In[72]:


# EXTRA TREES CLASSIFIER FEATURE SELECTION
# gathering testing and training data
feature_list_etc=[12, 13, 27, 28, 58, 63, 77, 78, 108, 109]
x1_train, x1_test, y1_train, y1_test = train_test_split(dataset.iloc[:,feature_list_etc], dataset_L.drop('Unnamed: 0', axis=1))


# In[73]:


# EXTRA TREES CLASSIFIER FEATURE SELECTION
# running logistic regression and training the model using fit with the training data
logreg1 = LogisticRegression(multi_class='ovr', solver='lbfgs', n_jobs=11)
logreg1.fit(x1_train, y1_train.values.ravel())


# In[74]:


# EXTRA TREES CLASSIFIER FEATURE SELECTION
# predicting the training and test data

y1_tr_pred = logreg1.predict(x1_train)
y1_pred = logreg1.predict(x1_test)


# In[94]:


# EXTRA TREES CLASSIFIER FEATURE SELECTION
# finding training and testing mse for the predictions found in the previous lines of code

print("Training MSE = ", mse(y1_tr_pred,y1_train))
print("Testing MSE = ", mse(y1_pred,y1_test))


# In[75]:


# EXTRA TREES CLASSIFIER FEATURE SELECTION
# finding the classification reports and accuracy scores for the training and testing datasets

print("\nTRAINING DATA:")
print(classification_report(y1_train, y1_tr_pred, target_names=['not malicous', 'malicous']))
print(metrics.accuracy_score(y1_train, y1_tr_pred))

print("\nTEST DATA:")
print(classification_report(y1_test, y1_pred, target_names=['not malicous', 'malicous']))
print(metrics.accuracy_score(y1_test, y1_pred))


# In[76]:


# EXTRA TREES CLASSIFIER FEATURE SELECTION
# confusion matrices for training and testing set

print("Number of 0's and 1's in y_train dataset:")
print(y1_train['x'].value_counts())
print("\nCONFUSION MATRIX FOR TRAINING SET: \n {}".format(confusion_matrix(y1_train, y1_tr_pred)))

print("\nNumber of 0's and 1's in y_test dataset:")
print(y1_test['x'].value_counts())
print("\nCONFUSION MATRIX FOR TESTING SET: \n {}".format(confusion_matrix(y1_test, y1_pred)))


# In[77]:


# SELECT K BEST FEATURE SELECTION
# gathering testing and training data

feature_list_skb=[12, 13, 27, 28, 63, 56, 77, 88, 108, 101]
x2_train, x2_test, y2_train, y2_test = train_test_split(dataset.iloc[:,feature_list_skb], dataset_L.drop('Unnamed: 0', axis=1))


# In[78]:


# SELECT K BEST FEATURE SELECTION
# running logistic regression and training the model using fit with the training data

logreg2 = LogisticRegression(multi_class='ovr', solver='lbfgs', n_jobs=11)
logreg2.fit(x2_train, y2_train.values.ravel())


# In[90]:


# SELECT K BEST FEATURE SELECTION
# predicting the training and test data

y2_tr_pred = logreg2.predict(x2_train)
y2_pred = logreg2.predict(x2_test)


# In[93]:


# SELECT K BEST FEATURE SELECTION
# finding training and testing mse for the predictions found in the previous lines of code

print("Training MSE = ", mse(y2_tr_pred,y2_train))
print("Testing MSE = ", mse(y2_pred,y2_test))


# In[92]:


# SELECT K BEST FEATURE SELECTION
# finding the classification reports and accuracy scores for the training and testing datasets

print("\nTRAINING DATA:")
print(classification_report(y2_train, y2_tr_pred, target_names=['not malicous', 'malicous']))
print(metrics.accuracy_score(y2_train, y2_tr_pred))

print("\nTEST DATA:")
print(classification_report(y2_test, y2_pred, target_names=['not malicous', 'malicous']))
print(metrics.accuracy_score(y2_test, y2_pred))


# In[85]:


# SELECT K BEST FEATURE SELECTION
# confusion matrices for training and testing set

print("Number of 0's and 1's in y_train dataset:")
print(y2_train['x'].value_counts())
print("\nCONFUSION MATRIX FOR TRAINING SET: \n {}".format(confusion_matrix(y1_train, y1_tr_pred)))

print("\nNumber of 0's and 1's in y_test dataset:")
print(y2_test['x'].value_counts())
print("\nCONFUSION MATRIX FOR TESTING SET: \n {}".format(confusion_matrix(y1_test, y1_pred)))


# In[ ]:




