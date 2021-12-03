#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Part B) Dataset Visualization

import os
import tarfile
import urllib

import pandas as pd

#PATH = os.path.join("datasets", "arp_mitm")
PATH = "Dataset"
FILE = "mirai.pcap.csv"
L_FILE = "mirai_labels.csv"

if(os.path.exists(PATH + '\\' + FILE)):
    print("Dataset Exists")
else:
    print("Dataset isn't available")
    
csv_path = os.path.join(PATH, FILE)
dataset = pd.read_csv(csv_path, header=None) 

#L_csv_path = os.path.join(PATH, L_FILE)
#dataset_L = pd.read_csv(L_csv_path, header=None, dtype={"": int, "x": int})  

print(dataset.head())
print(dataset.info())
print(dataset.describe())

#print(dataset_L.head())
#print(dataset_L.info())
#print(dataset_L.describe())


# In[5]:


L_csv_path = os.path.join(PATH, L_FILE)
dataset_L = pd.read_csv(L_csv_path, dtype={"": int, "x": 'float64'})  

print(dataset_L.head())
print(dataset_L.info())
print(dataset_L.describe())


# In[6]:


import matplotlib.pyplot as plt

#23 features per timegroup: 

# Bandwidth of outbound traffic (packet size)
    # srcMAC-IP(packet's source MAC and IP addr): 
        #---->mew_i[mean], sigma_i[standard deviation]
    
    # srcIP(packet source IP): 
        #---->mew_i[mean], sigma_i[standard deviation]
        
    # channel(packet's source and destination IP):
        #---->mew_i[mean], sigma_i[standard deviation]
        
    # socket(packet's source and destination address):
        #---->mew_i[mean], sigma_i[standard deviation]

# Bandwidth of outbound and inbound traffic together (packet size)
    # channel(packet's source and destination IP):
        #---->||s_i,s_j|| [magnitude], R_sisj [radius], cov_sisj [covariance], P_sisj [correlation coefficient]
        
    # socket(packet's source and destination address):
        #---->||s_i,s_j|| [magnitude], R_sisj [radius], cov_sisj [covariance], P_sisj [correlation coefficient]
    
# Packet rate of outbound traffic (packet count)
    # srcMAC-IP(packet's source MAC and IP addr): 
        #---->w_i[weight]
        
    # srcIP(packet source IP):
        #---->w_i[weight]
        
    # channel(packet's source and destination IP):
        #---->w_i[weight]    
    
    # socket(packet's source and destination address):
        #---->w_i[weight]    
        
    
# Inter-packet delays of outbound traffic (packet jitter)
    # channel(packet's source and destination IP): 
        #---->mew_i[mean], sigma_i[standard deviation],w_i[weight]


'''100ms time window data'''
dataset[dataset.columns[0:22]].hist(bins=50, figsize=(60,40))
plt.show()


# In[4]:


'''500ms time window data'''



# In[5]:





# In[6]:



# In[7]:



# In[8]:


import numpy as np

shuffled_indices = np.random.permutation(len(dataset))
test_set_size = int(len(dataset) * 0.20)
test_indices = shuffled_indices[:test_set_size]
train_indices = shuffled_indices[test_set_size:]
dataset.iloc[train_indices]
dataset.iloc[test_indices]

train_set = dataset.iloc[train_indices]
test_set = dataset.iloc[test_indices]

print(train_set.head())
print(test_set.head())


# In[25]:


'''
from sklearn.model_selection import train_test_split

train, test = train_test_split(dataset, test_size=0.2, random_state=42) # need something simple for classification
print(train.head())
print(test.head())
'''

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(dataset, dataset_L):
    strat_train_set = dataset.loc[train_index]
    train_zero = strat_train_set.loc[dataset_L == 0]
    train_one = strat_train_set.loc[dataset_L == 1]
    
    strat_train_L_set = dataset_L.loc[train_index]
    train_L_zero = strat_train_L_set.loc[dataset_L == 0]
    train_L_one = strat_train_L_set.loc[dataset_L == 1]
    
    strat_test_set = dataset.loc[test_index]
    test_zero = strat_test_set.loc[dataset_L == 0]
    test_one =strat_test_set.loc[dataset_L == 1]
    
    strat_test_L_set = dataset_L.loc[test_index]
    test_L_zero = strat_test_L_set.loc[dataset_L == 0]
    test_L_one = strat_test_L_set.loc[dataset_L == 1]
    
print("done")


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
'''
plt.scatter(train_zero[0].values, train_zero[1].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
plt.title("srcMAC-IP (non-malicious)")
plt.xlabel("mew_i[mean]")
plt.ylabel("sigma_i[standard deviation]")
plt.show()

plt.scatter(train_one[0].values, train_one[1].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
plt.title("srcMAC-IP (malicious)")
plt.xlabel("mew_i[mean]")
plt.ylabel("sigma_i[standard deviation]")
plt.show()
'''
print("Bandwidth of outbound traffic (packet size)")

f, a = plt.subplots(4, 2)

f.set_figheight(20)
f.set_figwidth(20)

a[0][0].scatter(train_zero[0].values, train_zero[1].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[0][0].set_title("srcMAC-IP (non-malicious) [100ms]")
a[0][0].set_xlabel("mew_i[mean]")
a[0][0].set_ylabel("sigma_i[standard deviation]")

a[0][1].scatter(train_one[0].values, train_one[1].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[0][1].set_title("srcMAC-IP (malicious) [100ms]")
a[0][1].set_xlabel("mew_i[mean]")
a[0][1].set_ylabel("sigma_i[standard deviation]")

a[1][0].scatter(train_zero[2].values, train_zero[3].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[1][0].set_title("srcIP (non-malicious) [100ms]")
a[1][0].set_xlabel("mew_i[mean]")
a[1][0].set_ylabel("sigma_i[standard deviation]")

a[1][1].scatter(train_one[2].values, train_one[3].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[1][1].set_title("srcIP (malicious) [100ms]")
a[1][1].set_xlabel("mew_i[mean]")
a[1][1].set_ylabel("sigma_i[standard deviation]")

a[2][0].scatter(train_zero[4].values, train_zero[5].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[2][0].set_title("srcIP (non-malicious) [100ms]")
a[2][0].set_xlabel("mew_i[mean]")
a[2][0].set_ylabel("sigma_i[standard deviation]")

a[2][1].scatter(train_one[4].values, train_one[5].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[2][1].set_title("srcIP (malicious) [100ms]")
a[2][1].set_xlabel("mew_i[mean]")
a[2][1].set_ylabel("sigma_i[standard deviation]")

a[3][0].scatter(train_zero[6].values, train_zero[7].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[3][0].set_title("socket (non-malicious) [100ms]")
a[3][0].set_xlabel("mew_i[mean]")
a[3][0].set_ylabel("sigma_i[standard deviation]")

a[3][1].scatter(train_one[6].values, train_one[7].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[3][1].set_title("socket (malicious) [100ms]")
a[3][1].set_xlabel("mew_i[mean]")
a[3][1].set_ylabel("sigma_i[standard deviation]")

f.tight_layout()
plt.show()


# In[11]:


print("Bandwidth of outbound and inbound traffic together (packet size)")

f, a = plt.subplots(6, 2)

l1 = "||s_i,s_j|| [magnitude]"
l2 = "R_sisj [radius]"
l3 = "cov_sisj [covariance]"
l4 = "P_sisj [correlation coefficient]"

f.set_figheight(20)
f.set_figwidth(20)

a[0][0].scatter(train_zero[8].values, train_zero[9].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[0][0].set_title("channel (non-malicious) [100ms]")
a[0][0].set_xlabel(l1)
a[0][0].set_ylabel(l2)

a[0][1].scatter(train_one[8].values, train_one[9].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[0][1].set_title("channel (malicious) [100ms]")
a[0][1].set_xlabel(l1)
a[0][1].set_ylabel(l2)

a[1][0].scatter(train_zero[8].values, train_zero[10].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[1][0].set_title("channel (non-malicious) [100ms]")
a[1][0].set_xlabel(l1)
a[1][0].set_ylabel(l3)

a[1][1].scatter(train_one[8].values, train_one[10].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[1][1].set_title("channel (malicious) [100ms]")
a[1][1].set_xlabel(l1)
a[1][1].set_ylabel(l3)

a[2][0].scatter(train_zero[8].values, train_zero[11].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[2][0].set_title("channel (non-malicious) [100ms]")
a[2][0].set_xlabel(l1)
a[2][0].set_ylabel(l4)

a[2][1].scatter(train_one[8].values, train_one[11].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[2][1].set_title("channel (malicious) [100ms]")
a[2][1].set_xlabel(l1)
a[2][1].set_ylabel(l4)

a[3][0].scatter(train_zero[9].values, train_zero[10].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[3][0].set_title("channel (non-malicious) [100ms]")
a[3][0].set_xlabel(l2)
a[3][0].set_ylabel(l3)

a[3][1].scatter(train_one[9].values, train_one[10].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[3][1].set_title("channel (malicious) [100ms]")
a[3][1].set_xlabel(l2)
a[3][1].set_ylabel(l3)

a[4][0].scatter(train_zero[9].values, train_zero[11].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[4][0].set_title("channel (non-malicious) [100ms]")
a[4][0].set_xlabel(l2)
a[4][0].set_ylabel(l4)

a[4][1].scatter(train_one[9].values, train_one[11].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[4][1].set_title("channel (malicious) [100ms]")
a[4][1].set_xlabel(l2)
a[4][1].set_ylabel(l4)

a[5][0].scatter(train_zero[10].values, train_zero[11].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[5][0].set_title("channel (non-malicious) [100ms]")
a[5][0].set_xlabel(l3)
a[5][0].set_ylabel(l4)

a[5][1].scatter(train_one[9].values, train_one[11].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[5][1].set_title("channel (malicious) [100ms]")
a[5][1].set_xlabel(l3)
a[5][1].set_ylabel(l4)

f.tight_layout()
plt.show()
    


# In[12]:


print("Bandwidth of outbound and inbound traffic together (packet size)")

f, a = plt.subplots(6, 2)

l1 = "||s_i,s_j|| [magnitude]"
l2 = "R_sisj [radius]"
l3 = "cov_sisj [covariance]"
l4 = "P_sisj [correlation coefficient]"

f.set_figheight(20)
f.set_figwidth(20)

a[0][0].scatter(train_zero[12].values, train_zero[13].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[0][0].set_title("socket (non-malicious) [100ms]")
a[0][0].set_xlabel(l1)
a[0][0].set_ylabel(l2)

a[0][1].scatter(train_one[12].values, train_one[13].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[0][1].set_title("socket (malicious) [100ms]")
a[0][1].set_xlabel(l1)
a[0][1].set_ylabel(l2)

a[1][0].scatter(train_zero[12].values, train_zero[14].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[1][0].set_title("socket (non-malicious) [100ms]")
a[1][0].set_xlabel(l1)
a[1][0].set_ylabel(l3)

a[1][1].scatter(train_one[12].values, train_one[14].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[1][1].set_title("socket (malicious) [100ms]")
a[1][1].set_xlabel(l1)
a[1][1].set_ylabel(l3)

a[2][0].scatter(train_zero[12].values, train_zero[15].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[2][0].set_title("socket (non-malicious) [100ms]")
a[2][0].set_xlabel(l1)
a[2][0].set_ylabel(l4)

a[2][1].scatter(train_one[12].values, train_one[15].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[2][1].set_title("socket (malicious) [100ms]")
a[2][1].set_xlabel(l1)
a[2][1].set_ylabel(l4)

a[3][0].scatter(train_zero[13].values, train_zero[14].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[3][0].set_title("socket (non-malicious) [100ms]")
a[3][0].set_xlabel(l2)
a[3][0].set_ylabel(l3)

a[3][1].scatter(train_one[13].values, train_one[14].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[3][1].set_title("socket (malicious) [100ms]")
a[3][1].set_xlabel(l2)
a[3][1].set_ylabel(l3)

a[4][0].scatter(train_zero[13].values, train_zero[15].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[4][0].set_title("socket (non-malicious) [100ms]")
a[4][0].set_xlabel(l2)
a[4][0].set_ylabel(l4)

a[4][1].scatter(train_one[13].values, train_one[15].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[4][1].set_title("socket (malicious) [100ms]")
a[4][1].set_xlabel(l2)
a[4][1].set_ylabel(l4)

a[5][0].scatter(train_zero[14].values, train_zero[15].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[5][0].set_title("socket (non-malicious) [100ms]")
a[5][0].set_xlabel(l3)
a[5][0].set_ylabel(l4)

a[5][1].scatter(train_one[14].values, train_one[15].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[5][1].set_title("socket (malicious) [100ms]")
a[5][1].set_xlabel(l3)
a[5][1].set_ylabel(l4)

f.tight_layout()
plt.show()
    


# In[13]:


print("Packet rate of outbound traffic (packet count)")

f, a = plt.subplots(6, 2)

f.set_figheight(20)
f.set_figwidth(20)

a[0][0].scatter(train_zero[16].values, train_zero[17].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[0][0].set_title("socket (non-malicious) [100ms]")
a[0][0].set_xlabel("w_i[weight]")
a[0][0].set_ylabel("w_i[weight]")

a[0][1].scatter(train_one[16].values, train_one[17].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[0][1].set_title("socket (malicious) [100ms]")
a[0][1].set_xlabel("w_i[weight]")
a[0][1].set_ylabel("w_i[weight]")

a[1][0].scatter(train_zero[16].values, train_zero[18].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[1][0].set_title("socket (non-malicious) [100ms]")
a[1][0].set_xlabel("w_i[weight]")
a[1][0].set_ylabel("w_i[weight]")

a[1][1].scatter(train_one[16].values, train_one[18].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[1][1].set_title("socket (malicious) [100ms]")
a[1][1].set_xlabel("w_i[weight]")
a[1][1].set_ylabel("w_i[weight]")

a[2][0].scatter(train_zero[16].values, train_zero[19].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[2][0].set_title("socket (non-malicious) [100ms]")
a[2][0].set_xlabel("w_i[weight]")
a[2][0].set_ylabel("w_i[weight]")

a[2][1].scatter(train_one[16].values, train_one[19].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[2][1].set_title("socket (malicious) [100ms]")
a[2][1].set_xlabel("w_i[weight]")
a[2][1].set_ylabel("w_i[weight]")

a[3][0].scatter(train_zero[17].values, train_zero[18].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[3][0].set_title("socket (non-malicious) [100ms]")
a[3][0].set_xlabel("w_i[weight]")
a[3][0].set_ylabel("w_i[weight]")

a[3][1].scatter(train_one[17].values, train_one[18].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[3][1].set_title("socket (malicious) [100ms]")
a[3][1].set_xlabel("w_i[weight]")
a[3][1].set_ylabel("w_i[weight]")

a[4][0].scatter(train_zero[17].values, train_zero[19].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[4][0].set_title("socket (non-malicious) [100ms]")
a[4][0].set_xlabel("w_i[weight]")
a[4][0].set_ylabel("w_i[weight]")

a[4][1].scatter(train_one[17].values, train_one[19].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[4][1].set_title("socket (malicious) [100ms]")
a[4][1].set_xlabel("w_i[weight]")
a[4][1].set_ylabel("w_i[weight]")

a[5][0].scatter(train_zero[18].values, train_zero[19].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[5][0].set_title("socket (non-malicious) [100ms]")
a[5][0].set_xlabel("w_i[weight]")
a[5][0].set_ylabel("w_i[weight]")

a[5][1].scatter(train_one[18].values, train_one[19].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[5][1].set_title("socket (malicious) [100ms]")
a[5][1].set_xlabel("w_i[weight]")
a[5][1].set_ylabel("w_i[weight]")

f.tight_layout()
plt.show()
    


# In[14]:


print("Inter-packet delays of outbound traffic (packet jitter)")

f, a = plt.subplots(3, 2)

f.set_figheight(20)
f.set_figwidth(20)

a[0][0].scatter(train_zero[20].values, train_zero[21].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[0][0].set_title("socket (non-malicious) [100ms]")
a[0][0].set_xlabel("mew_i[mean]")
a[0][0].set_ylabel("sigma_i[standard deviation]")

a[0][1].scatter(train_one[20].values, train_one[21].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[0][1].set_title("socket (malicious) [100ms]")
a[0][1].set_xlabel("mew_i[mean]")
a[0][1].set_ylabel("sigma_i[standard deviation]")

a[1][0].scatter(train_zero[20].values, train_zero[22].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[1][0].set_title("socket (non-malicious) [100ms]")
a[1][0].set_xlabel("mew_i[mean]")
a[1][0].set_ylabel("w_i[weight]")

a[1][1].scatter(train_one[20].values, train_one[22].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[1][1].set_title("socket (malicious) [100ms]")
a[1][1].set_xlabel("mew_i[mean]")
a[1][1].set_ylabel("w_i[weight]")

a[2][0].scatter(train_zero[21].values, train_zero[22].values, alpha=0.4, color="blue", cmap=plt.get_cmap("jet"))
a[2][0].set_title("socket (non-malicious) [100ms]")
a[2][0].set_xlabel("sigma_i[standard deviation]")
a[2][0].set_ylabel("w_i[weight]")

a[2][1].scatter(train_one[21].values, train_one[22].values, alpha=0.4, color="red", cmap=plt.get_cmap("jet"))
a[2][1].set_title("socket (malicious) [100ms]")
a[2][1].set_xlabel("sigma_i[standard deviation]")
a[2][1].set_ylabel("w_i[weight]")

f.tight_layout()
plt.show()
    


# In[35]:


# The following shows the differences in histograms of data from malicious and safe packets
# The graphs show the relationship between malicious and safe packets and show that... a difference exists but it's very slim

for i in range(0, 23):
    plt.hist(train_zero[i].values, alpha=0.5, label='safe')
    plt.hist(train_one[i].values, alpha=0.5, label='malicious')
    plt.xlabel("Feature {}".format(i), size=14)
    plt.ylabel("Magnitude of Feature", size=14)
    plt.legend(loc="upper left")
    plt.show()


# In[36]:


for i in range(23, 46):
    plt.hist(train_zero[i].values, alpha=0.5, label='safe')
    plt.hist(train_one[i].values, alpha=0.5, label='malicious')
    plt.xlabel("Feature {}".format(i), size=14)
    plt.ylabel("Magnitude of Feature", size=14)
    plt.legend(loc="upper left")
    plt.show()


# In[37]:


for i in range(46, 69):
    plt.hist(train_zero[i].values, alpha=0.5, label='safe')
    plt.hist(train_one[i].values, alpha=0.5, label='malicious')
    plt.xlabel("Feature {}".format(i), size=14)
    plt.ylabel("Magnitude of Feature", size=14)
    plt.legend(loc="upper left")
    plt.show()


# In[38]:


for i in range(69, 92):
    plt.hist(train_zero[i].values, alpha=0.5, label='safe')
    plt.hist(train_one[i].values, alpha=0.5, label='malicious')
    plt.xlabel("Feature {}".format(i), size=14)
    plt.ylabel("Magnitude of Feature", size=14)
    plt.legend(loc="upper left")
    plt.show()


# In[39]:


for i in range(92, 115):
    plt.hist(train_zero[i].values, alpha=0.5, label='safe')
    plt.hist(train_one[i].values, alpha=0.5, label='malicious')
    plt.xlabel("Feature {}".format(i), size=14)
    plt.ylabel("Magnitude of Feature", size=14)
    plt.legend(loc="upper left")
    plt.show()


# In[ ]:




