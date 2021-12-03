#!/usr/bin/env python
# coding: utf-8

# In[41]:


#Part g) Fine-Tuning

import os
import tarfile
import urllib

import pandas as pd
from sklearn.metrics import mean_squared_error as mse


#reading ARP MitM dataset
PATH = "arp_mitm"

FILE = "ARP MitM_dataset-002.csv"
L_FILE = "ARP MitM_labels.csv"

#input data -> measured
csv_path = os.path.join(PATH, FILE)
dataset_filt = pd.read_csv(csv_path, header=None)  
#dataset_filt = dataset.dropna()
#display(dataset.head())
#display(dataset.info())
#display(dataset.describe())


# In[42]:


#Reading output data is the observation of malicous and none malicous 
csv_path_L = os.path.join(PATH, L_FILE)
dataset_L_filt = pd.read_csv(csv_path_L, dtype={"": int, "x": 'float64'})  


display(dataset_L_filt.head())
display(dataset_L_filt.info())
display(dataset_L_filt.describe())


# In[43]:


#counting how many malicous and non malicous was observed in the dataset
#dataset_L_filt.drop('Unnamed: 0', axis=1)
dataset_L_filt["x"].value_counts()


# In[48]:


#trying to look at the coleration between feature but clearly this does not help much since we have wait too much data.
#all the feature have some kind of coleration to one another as observe. This indicate that we cant fix one input while 
#changing the other inputs. 
correlation = dataset.corr()
correlation


# In[49]:


#Part g) Fine-Tuning

import numpy as np
import matplotlib as plt
import scipy.stats

dataset = None
dataset_L = None

#extracing all rows that have values that abide by Zscore < 2 standard deviations from the mean

z_score = np.abs(stats.zscore(dataset_filt))
dataset = dataset_filt[(z_score < 2).all(axis=1)]
dataset_L = dataset_L_filt[(z_score < 2).all(axis=1)]

print("original dataset:")
display(dataset.info())
display(dataset_L.info())
print("new with IQR outlier filtering:")
display(dataset_filt.info())
display(dataset_L_filt.info())


# In[50]:


# part h) accuracy scores

#LDA classification model, the classification metric is also show in the result print out as the measurement of model accuratecy

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn import metrics
from sklearn.model_selection import train_test_split

#splitting the dataset for the cross validation test on unseen testing data. 
#splitting ratio is 3:1 for training and testing
x_train, x_test, y_train, y_test = train_test_split( dataset, dataset_L, test_size=0.75, random_state=4)


lda = LinearDiscriminantAnalysis(solver='svd')
lda_training = lda.fit(x_train,(y_train['x']==1)).predict(x_train)
lda_testing = lda.fit(x_test,(y_test['x']==1)).predict(x_test)


print("\nTRAINING DATA:")
print(classification_report((y_train['x']==1), lda_training, target_names=['not malicous', 'malicous']))
print(metrics.accuracy_score((y_train['x']==1), lda_training))

print("\nTEST DATA:")
print(classification_report((y_test['x']==1), lda_testing, target_names=['not malicous', 'malicous']))
print(metrics.accuracy_score((y_test['x']==1), lda_testing))


# In[51]:


# part h) training mse and test mse

print("FOR MODEL WITH NO FEATURE SELECTION")
print("Training MSE = ", mse(lda_training,y_train['x'].values))
print("Testing MSE = ", mse(lda_testing,y_test['x'].values))


# In[59]:


# part h) Confusion Matrices

lda.fit(x_train,y_train['x'].values.ravel())

y_tr_pred = lda.predict(x_train)
y_pred = lda.predict(x_test)

print("Number of 0's and 1's in y_train dataset:")
print(y_train['x'].value_counts())
print("\nCONFUSION MATRIX FOR TRAINING SET: \n {}".format(confusion_matrix(y_train['x'].values.ravel(), y_tr_pred)))

print("\nNumber of 0's and 1's in y_test dataset:")
print(y_test['x'].value_counts())
print("\nCONFUSION MATRIX FOR TESTING SET: \n {}".format(confusion_matrix(y_test['x'].values.ravel(), y_pred)))


# In[62]:


# Part h) ROC curve

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

lda_roc = lda.predict_proba(x_test)

false_pos, true_pos, throwaway= roc_curve(y_test['x'].values.ravel(), lda_roc[:, 1])

plt.plot(false_pos, true_pos, label='ROC Curve')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title("ROC Curve for LDA Classifier Model\nWith no Feature Selection")
plt.legend()
plt.show()


# In[ ]:




