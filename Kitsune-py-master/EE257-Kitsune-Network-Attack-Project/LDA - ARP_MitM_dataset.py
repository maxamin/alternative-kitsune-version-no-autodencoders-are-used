#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
dataset = pd.read_csv(csv_path, header=None)  
dataset = dataset.dropna()
display(dataset.head())
display(dataset.info())
display(dataset.describe())


# In[3]:


#Reading output data is the observation of malicous and none malicous 
csv_path_L = os.path.join(PATH, L_FILE)
dataset_L = pd.read_csv(csv_path_L, dtype={"": int, "x": 'float64'})  


display(dataset_L.head())
display(dataset_L.info())
display(dataset_L.describe())


# In[4]:


#counting how many malicous and non malicous was observed in the dataset
dataset_L["x"].value_counts()


# In[5]:


#trying to look at the coleration between feature but clearly this does not help much since we have wait too much data.
#all the feature have some kind of coleration to one another as observe. This indicate that we cant fix one input while 
#changing the other inputs. 
correlation = dataset.corr()
correlation


# In[42]:


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
x_train, x_test, y_train, y_test = train_test_split( dataset, dataset_L, random_state=4)


lda = LinearDiscriminantAnalysis(solver='svd')
lda_training = lda.fit(x_train,(y_train['x']==1)).predict(x_train)
lda_testing = lda.fit(x_test,(y_test['x']==1)).predict(x_test)


print("\nTRAINING DATA:")
print(classification_report((y_train['x']==1), lda_training, target_names=['not malicous', 'malicous']))
print(metrics.accuracy_score((y_train['x']==1), lda_training))

print("\nTEST DATA:")
print(classification_report((y_test['x']==1), lda_testing, target_names=['not malicous', 'malicous']))
print(metrics.accuracy_score((y_test['x']==1), lda_testing))


# In[41]:


print("FOR MODEL WITH NO FEATURE SELECTION")
print("Training MSE = ", mse(lda_training,y_train['x'].values))
print("Testing MSE = ", mse(lda_testing,y_test['x'].values))


# In[9]:


#comment: from both QDA and LDA, the random splitting ratio 1:1 giving the best testing error while the training error
#barely change. This telling us that we might have been overfitting the model when having a bigger training data


# In[10]:


#feature selection done by ExtraTreesClassifier()
feature_select = [12, 13, 27, 28, 58, 63, 77, 79, 108, 109]

x=dataset.iloc[:,feature_select]
x1=dataset
y=dataset_L.drop('Unnamed: 0', axis=1)

display(x.head(5))
display(y.head(5))


# In[38]:


#LDA classification model with selected features by Extra Trees Classifier
#splitting ratio is 3:1 for training and testing
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.33, random_state=4)


lda = LinearDiscriminantAnalysis(solver='svd')
lda_training = lda.fit(x_train,(y_train['x']==1)).predict(x_train)
lda_testing = lda.fit(x_test,(y_test['x']==1)).predict(x_test)


print("\nTRAINING DATA:")
print(classification_report((y_train['x']==1), lda_training, target_names=['not malicous', 'malicous']))
print(metrics.accuracy_score((y_train['x']==1), lda_training))

print("\nTEST DATA:")
print(classification_report((y_test['x']==1), lda_testing, target_names=['not malicous', 'malicous']))
print(metrics.accuracy_score((y_test['x']==1), lda_testing))
confusion_matrix(y_test, lda_testing)


# In[39]:


print("FOR MODEL WITH EXTRA TREES CLASSIFIER FEATURE SELECTION")
print("Training MSE = ", mse(lda_training,y_train['x'].values))
print("Testing MSE = ", mse(lda_testing,y_test['x'].values))


# In[36]:


#LDA classification model with selected features by Select K Best Classifier
#splitting ratio is 3:1 for training and testing
fselect2=[12, 13, 27, 28, 63, 56, 77, 88, 108, 101]

x=dataset.iloc[:,fselect2]
x1=dataset
y=dataset_L.drop('Unnamed: 0', axis=1)

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.33, random_state=4)

lda = LinearDiscriminantAnalysis(solver='svd')
lda_training = lda.fit(x_train,(y_train['x']==1)).predict(x_train)
lda_testing = lda.fit(x_test,(y_test['x']==1)).predict(x_test)


print("\nTRAINING DATA:")
print(classification_report((y_train['x']==1), lda_training, target_names=['not malicous', 'malicous']))
print(metrics.accuracy_score((y_train['x']==1), lda_training))

print("\nTEST DATA:")
print(classification_report((y_test['x']==1), lda_testing, target_names=['not malicous', 'malicous']))
print(metrics.accuracy_score((y_test['x']==1), lda_testing))
confusion_matrix(y_test, lda_testing)


# In[37]:


print("FOR MODEL WITH SELECT K BEST CLASSIFIER FEATURE SELECTION")
print("Training MSE = ", mse(lda_training,y_train['x'].values))
print("Testing MSE = ", mse(lda_testing,y_test['x'].values))


# In[13]:


#the selection model from the extratreeclasscification is that working well with the LDA and QDA, the main reason
#for this is that we do not have enough data for training set. 


# In[14]:


#conclusion form the LDA and QDA
#1. the best model id QDA with fine tune splitting dataset between training and testing set as ratio 1:1
#2. QDA working better than LDA implied that the dataset have non-linear realtionship.
#3. QDA give best test error rate of  0.9788864450609919 and testing error 0.9786883661719642 which is very close to one.
# also the performence metrics indicate a very good result for both training and testing data sucj as presicion, recal, 
# and F1-score are all very hight (in range of .97 abd .98)

