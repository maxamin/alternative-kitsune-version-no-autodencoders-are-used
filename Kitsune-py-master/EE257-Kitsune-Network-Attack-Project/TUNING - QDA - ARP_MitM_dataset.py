#!/usr/bin/env python
# coding: utf-8

# In[12]:


#Part g) Fine-Tuning

import os
import tarfile
import urllib

import pandas as pd
from sklearn.metrics import mean_squared_error as mse

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn import metrics
from sklearn.model_selection import train_test_split


#reading ARP MitM dataset
PATH = "arp_mitm"

FILE = "ARP MitM_dataset-002.csv"
L_FILE = "ARP MitM_labels.csv"

#input data -> measured
csv_path = os.path.join(PATH, FILE)
dataset_filt = pd.read_csv(csv_path, header=None)  

dataset = dataset.dropna()
display(dataset.head())
display(dataset.info())
display(dataset.describe())


# In[6]:


#Reading output data is the observation of malicous and none malicous 
csv_path_L = os.path.join(PATH, L_FILE)

dataset_L_filt = pd.read_csv(csv_path_L, dtype={"": int, "x": 'float64'})  

display(dataset_L.head())
display(dataset_L.info())
display(dataset_L.describe())


# In[37]:


import numpy as np
import matplotlib as plt
import scipy.stats

dataset = None
dataset_L = None

#extracing all rows that have values that abide by Zscore < 2 standard deviations from the mean

z_score = np.abs(zscore(dataset_filt))
dataset = dataset_filt[(z_score < 2).all(axis=1)]
dataset_L = dataset_L_filt[(z_score < 2).all(axis=1)]

print("original dataset:")
display(dataset.info())
display(dataset_L.info())
print("new with IQR outlier filtering:")
display(dataset_filt.info())
display(dataset_L_filt.info())


# In[38]:


# part h) accuracy scores

# QDA modeling to see if non-linear model would fix better

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#splitting ratio is 85:15 for training and testing
x_train, x_test, y_train, y_test = train_test_split( dataset, dataset_L, test_size=0.15, random_state=4)


qda = QuadraticDiscriminantAnalysis()
qda_training = qda.fit(x_train,(y_train['x']==1)).predict(x_train)
qda_testing = qda.fit(x_test,(y_test['x']==1)).predict(x_test)


print("\nTRAINING DATA:")
print(classification_report((y_train['x']==1), qda_training, target_names=['not malicous', 'malicous']))
print(metrics.accuracy_score((y_train['x']==1), qda_training))

print("\nTEST DATA:")
print(classification_report((y_test['x']==1), qda_testing, target_names=['not malicous', 'malicous']))
print(metrics.accuracy_score((y_test['x']==1), qda_testing))


# In[39]:


# part h) training mse and test mse

print("FOR MODEL WITH NO FEATURE SELECTION")
print("Training MSE = ", mse(qda_training,y_train['x'].values))
print("Testing MSE = ", mse(qda_testing,y_test['x'].values))


# In[40]:


# part h) Confusion Matrices

qda.fit(x_train,y_train['x'].values.ravel())

y_tr_pred = qda.predict(x_train)
y_pred = qda.predict(x_test)

print("Number of 0's and 1's in y_train dataset:")
print(y_train['x'].value_counts())
print("\nCONFUSION MATRIX FOR TRAINING SET: \n {}".format(confusion_matrix(y_train['x'].values.ravel(), y_tr_pred)))

print("\nNumber of 0's and 1's in y_test dataset:")
print(y_test['x'].value_counts())
print("\nCONFUSION MATRIX FOR TESTING SET: \n {}".format(confusion_matrix(y_test['x'].values.ravel(), y_pred)))


# In[41]:


# Part h) ROC curve

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

qda_roc = qda.predict_proba(x_test)

false_pos, true_pos, throwaway= roc_curve(y_test['x'].values.ravel(), qda_roc[:, 1])

plt.plot(false_pos, true_pos, label='ROC Curve')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title("ROC Curve for QDA Classifier Model\nWith no Feature Selection")
plt.legend()
plt.show()


# In[42]:


quartile_one = dataset_filt.quantile(0.05)
quartile_three = dataset_filt.quantile(0.95)
innerquartilerange = quartile_three - quartile_one

dataset_iqr = dataset_filt[~((dataset_filt < (quartile_one - 1.5 * innerquartilerange)) |(dataset_filt > (quartile_three + 1.5 * innerquartilerange))).any(axis=1)]
dataset_L_iqr = dataset_L_filt[~((dataset_filt < (quartile_one - 1.5 * innerquartilerange)) |(dataset_filt > (quartile_three + 1.5 * innerquartilerange))).any(axis=1)]

feature_select = [12, 13, 27, 28, 63, 56, 77, 88, 108, 101]

x=dataset_iqr.iloc[:,feature_select]
x1=dataset_iqr
y=dataset_L_iqr.drop('Unnamed: 0', axis=1)

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.33, random_state=4)

qda = QuadraticDiscriminantAnalysis()
qda_training = qda.fit(x_train,(y_train['x']==1)).predict(x_train)
qda_testing = qda.fit(x_test,(y_test['x']==1)).predict(x_test)


print("\nTRAINING DATA:")
print(classification_report((y_train['x']==1), qda_training, target_names=['not malicous', 'malicous']))
print(metrics.accuracy_score((y_train['x']==1), qda_training))

print("\nTEST DATA:")
print(classification_report((y_test['x']==1), qda_testing, target_names=['not malicous', 'malicous']))
print(metrics.accuracy_score((y_test['x']==1), qda_testing))


# In[43]:


print("FOR MODEL WITH SELECT K BEST FEATURE SELECTION")
print("Training MSE = ", mse(qda_training,y_train['x'].values))
print("Testing MSE = ", mse(qda_testing,y_test['x'].values))


# In[ ]:




