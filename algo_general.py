# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 22:19:13 2021

@author: Laptop Center
"""
import time
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import  RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy.stats import norm
from matplotlib import pyplot as plt
import warnings
from dtreeviz.trees import dtreeviz, rtreeviz_univar  # remember to load the package
warnings.filterwarnings("ignore")
feature_list_skb=["frame.time_epoch","frame.len","tcp.srcport","tcp.dstport","udp.srcport","udp.dstport","icmp.type","icmp.code","arp.opcode"]


os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin\\'
def fratello(RMSEs,FMgrace,ADgrace,start,stop,algo_name="",tl=[]):
	print("Complete. Time elapsed: " + str(stop - start))

	# Here we demonstrate how one can fit the RMSE scores to a log-normal distribution (useful for finding/setting a cutoff threshold \phi)
	benignSample = np.log(RMSEs[0:FMgrace+ADgrace])
	logProbs = norm.logsf(np.log(RMSEs), np.mean(benignSample), np.std(benignSample))

	# plot the RMSE anomaly scores
	print("Plotting results")

	plt.figure(figsize=(10, 5))
	print(tl)
	fig = plt.scatter([x for x in tl], RMSEs[0:], s=3,
					  c=logProbs[0:FMgrace+ADgrace], cmap='RdYlGn')
	plt.yscale("log")
	plt.title("Anomaly Scores from Kitsune's Execution Phase [["+str(algo_name)+"]]")
	plt.ylabel("RMSE (log scaled)")
	plt.xlabel("Time elapsed [S]")
	figbar = plt.colorbar()
	figbar.ax.set_ylabel('Log Probability\n ', rotation=270)
	plt.show()

def calculate_rmse(Yact,Ypred):
	return mean_squared_error(Yact, Ypred, squared=False)
PATH = "Dataset"
FILE = "mirai.pcap.csv"
L_FILE = "mirai_labels.csv"
cnames=(pd.read_csv('dataset/names.csv').columns.tolist())
csv_path = os.path.join(PATH, FILE)
dataset = pd.read_csv(csv_path,names=cnames)
if(len(dataset.index)>=50000):
	FMgrace=round((5000/(len(dataset.index))))
	ADgrace=round((50000/(len(dataset.index))))
else:
	FMgrace=0.2
	ADgrace=0.8
print(dataset.head())
print(dataset.info())
print(dataset.describe())

# In[77]:


# gathering classified output data for dataset
L_csv_path = os.path.join(PATH, L_FILE)
dataset_L = pd.read_csv(L_csv_path, dtype=int, nrows=len(dataset.index))
dataset=dataset.replace(np.nan, 0, regex=True)
if(len(open(L_csv_path, 'r').readlines()[0])<=0):
	python("cannot accept empty labels dataset")
	exit()
if(len(open(L_csv_path, 'r').readlines()[0]) >= 3):
	dataset_L = pd.read_csv(L_csv_path,names=[0,1] , nrows=len(dataset.index))
	dataset_L[1]=dataset_L[1][1:]
	dataset_L=pd.DataFrame(dataset_L[1])
	dataset_L.columns = [''] * len(dataset_L.columns)
	dataset_L=dataset_L.replace(np.nan, 0, regex=True)
	dataset_L=dataset_L.astype('int32')
else:
	dataset_L = pd.read_csv(L_csv_path, dtype=int, nrows=len(dataset.index))
print(dataset_L.head())
print(dataset_L.info())
print(dataset_L.describe())


# In[78]:


# EXTRA TREES CLASSIFIER FEATURE SELECTION
# gathering testing and training data
dataset=dataset.replace(np.nan, 0, regex=True)
dataset_L=dataset_L.replace(np.nan, 0, regex=True)
X_train, X_test, Y_train, Y_test = train_test_split(dataset.loc[:,feature_list_skb], dataset_L, train_size=FMgrace,test_size=ADgrace,random_state=0)
# In[79]:


# part g) fine tuning

# EXTRA TREES CLASSIFIER FEATURE SELECTION
# running logistic regression and training the model using fit with the training data

# MODIFIED "C" OR INVERSE REGULARIZATION STRENGTH

def LOR():
	logreg1 = LogisticRegression(multi_class='ovr', solver='lbfgs', n_jobs=11, C=20.0)
	logreg1.fit(X_train, Y_train.values.ravel())
	Y_tr_pred = logreg1.predict(X_train)
	Y_pred = logreg1.predict(X_test)
	print("\nTRAINING DATA:")
	print(classification_report(Y_train, Y_tr_pred, target_names=['not malicous', 'malicous']))
	print(metrics.f1_score(Y_train, Y_tr_pred,average='weighted', labels=np.unique(Y_tr_pred),zero_division=1))
	print("\nTEST DATA:")
	print(classification_report(Y_test, Y_pred, target_names=['not malicous', 'malicous']))
	print(metrics.f1_score(Y_test, Y_pred,average='weighted', labels=np.unique(Y_pred),zero_division=1))
	print("Number of 0's and 1's in Y_train dataset:")
	print(Y_train['0'].value_counts())
	print("\nCONFUSION MATRIX FOR TRAINING SET: \n {}".format(confusion_matrix(Y_train, Y_tr_pred)))
	print("\nNumber of 0's and 1's in Y_test dataset:")
	print(Y_test['0'].value_counts())
	print("\nCONFUSION MATRIX FOR TESTING SET: \n {}".format(confusion_matrix(Y_test, Y_pred)))
	return calculate_rmse(Y_train,Y_tr_pred)
#plotit(LOR)
# In[83]:


# SELECT K BEST FEATURE SELECTION
# gathering testing and training data

# In[84]:


# part g) fine tuning

# SELECT K BEST FEATURE SELECTION
# running logistic regression and training the model using fit with the training data

# MODIFIED "C" OR INVERSE REGULARIZATION STRENGTH

#########################################################################
# Decision Tree 
def DT():
	start_time = time.time()
	tl=[]
	depth = []
	Rmses=[]
	for i in range(1, 200):
		l_s=time.time()
		clfd = tree.DecisionTreeClassifier(max_depth=i)
		clfd.fit(X_train, Y_train.values.ravel())
		# Perform 7-fold cross validation
		scores = cross_val_score(estimator=clfd, X=X_train, y=Y_train.values.ravel(), cv=7, n_jobs=4)
		depth.append(i)
		Rmses.append(scores.mean())
		l_e=time.time()
		tl.append(l_e-l_s)


	end_time = time.time()
	fratello(Rmses, depth[-1], 0, start_time,end_time,"DecisionTree",tl)
	log=(str("Training time: ")+ str(end_time-start_time)+"\n")
	start_time = time.time()
	Y_test_pred = clfd.predict(X_train)
	end_time = time.time()
	log+=str("Testing time: "+ str(end_time-start_time)+"\n")
	log+=str("Train score is:"+ str(clfd.score(X_train, Y_train))+"\n")
	log += str(("Test score is:" + str(clfd.score(X_test, Y_test))+"\n"))
	log+=str("DT Last Classifier RMSE = " + str(calculate_rmse(Y_train, Y_test_pred))+"\n")
	if ((336620 > ((len(dataset.index))*ADgrace))):
		try:
			viz = dtreeviz(clfd, X_train, pd.DataFrame(Y_test_pred).values.ravel(),target_name="Risk level",class_names=["Normal","Malicious"],feature_names=list(X_train.columns.values),orientation='LR' ,max_X_features_LR=100,title=log,colors={'scatter_marker': '#00ff00'})
			viz.view()
		except:
			print("Failed to generate bitmap tree cause the dataset is way too large")
	else:
		print("you are using a big dataset to run script you can dummify the if statement and pass to the dtreeviz but better you have a good computing power :)")

	return calculate_rmse(Y_train,Y_test_pred)
#########################################################################
#Random Forest
def RF():
	t_start_time = time.time()
	tl = []
	Rmses = []
	depth = []
	for i in range(1, 200):
		l_s = time.time()
		clfr = RandomForestClassifier(max_depth=i)
		clfr.fit(X_train, Y_train.values.ravel())
		# Perform 7-fold cross validation
		scores = cross_val_score(estimator=clfr, X=X_train, y=Y_train.values.ravel(), cv=7, n_jobs=4)
		depth.append(i)
		Rmses.append(scores.mean())
		l_e=time.time()
		tl.append(l_e-l_s)
	t_end_time = time.time()
	log = (str("Training time: ") + str(t_end_time - t_start_time) + "\n")
	fratello(Rmses, depth[-1], 0, t_start_time, t_end_time,"Random Forest",tl)
	start_time = time.time()
	Y_test_pred = clfr.predict(X_train)
	end_time = time.time()
	log+=str("Testing time: "+ str(end_time-start_time)+"\n")
	log+=str("Train score is:"+ str(clfr.score(X_train, Y_train))+"\n")
	log += str(("Test score is:" + str(clfr.score(X_test, Y_test)))+"\n")
	log+=str("RF Last Classifier rmse = " + str(calculate_rmse(Y_train, Y_test_pred))+"\n")
	if ((336620 > ((len(dataset.index))*ADgrace))):
		try:
			vizrf = dtreeviz(clfr.estimators_[0], X_train, pd.DataFrame(Y_test_pred).values.ravel(),class_names=["Normal","Malicious"],target_name="Risk level",feature_names=list(X_train.columns.values),orientation='LR' ,max_X_features_LR=None,title=log,colors={'scatter_marker': '#00ff00'})
			vizrf.view()
		except:
			print("Failed to generate bitmap tree cause the dataset is way too large")
	else:
		print("you are using a big dataset to run script you can dummify the if statement and pass to the dtreeviz but better you have a good computing power :)")

	return calculate_rmse(Y_train,Y_test_pred)
########################################################################
#ET
def ET():
	t_start_time = time.time()
	Rmses=[]
	tl=[]
	depth = []
	for i in range(1, 200):
		try:
			l_s=time.time()
			clfg = tree.ExtraTreeClassifier(max_depth=i)
			clfg.fit(X_train, Y_train.values.ravel())
			# Perform 7-fold cross validation
			scores = cross_val_score(estimator=clfg, X=X_train, y=Y_train.values.ravel(), cv=7, n_jobs=4)
			depth.append(i)
			Rmses.append(scores.mean())
			l_e=time.time()
			tl.append(l_e-l_s)
		except:
			break
	t_end_time = time.time()
	fratello(Rmses, depth[-1], 0, t_start_time, t_end_time,"extra tree",tl)
	start_time = time.time()
	Y_test_pred = clfg.predict(X_train)
	end_time = time.time()
	log = (str("Training time: ") + str(end_time - start_time) + "\n")
	log+=str("Testing time: "+ str(end_time-start_time)+"\n")
	log+=str("Train score is:"+ str(clfg.score(X_train, Y_train))+"\n")
	log += str(("Test score is:" + str(clfg.score(X_test, Y_test)))+"\n")
	log+=str("ET Last Classifier rmse = " + str(calculate_rmse(Y_train, Y_test_pred))+"\n")
	if ((336620 > ((len(dataset.index))*ADgrace))):
		try:
			vizrET = dtreeviz(clfg, X_train, pd.DataFrame(Y_test_pred).values.ravel(),target_name="Risk level",class_names=["Normal","Malicious"],feature_names=list(X_train.columns.values),orientation='LR' ,max_X_features_LR=None,title=log,colors={'scatter_marker': '#00ff00'})
			vizrET.view()
		except:
			print("Failed to generate bitmap tree cause the dataset is way too large")
	else:
		print("you are using a big dataset to run script you can dummify the if statement and pass to the dtreeviz but better you have a good computing power :)")
	return calculate_rmse(Y_train,Y_test_pred)
#######################################################################

ET()
DT()
RF()



