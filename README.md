# Kitsune
## New theme Trees Magic and more

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1024px-Python-logo-notext.svg.png" style="height:200px;width:200px;"/>

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

What is Kitsune?

Neural networks have become an increasingly popular solution for network intrusion detection systems (NIDS). Their capability of learning complex patterns and behaviors make them a suitable solution for differentiating between normal traffic and network attacks. However, a drawback of neural networks is the amount of resources needed to train them. Many network gateways and routers devices, which could potentially host an NIDS, simply do not have the memory or processing power to train and sometimes even execute such models. More importantly, the existing neural network solutions are trained in a supervised manner. Meaning that an expert must label the network traffic and update the model manually from time to time.

Kitsune is a novel ANN-based NIDS which is online, unsupervised, and efficient. A Kitsune, in Japanese folklore, is a mythical fox-like creature that has a number of tails, can mimic different forms, and whose strength increases with experience. Similarly, Kitsune has an ensemble of small neural networks (autoencoders), which are trained to mimic (reconstruct) network traffic patterns, and whose performance incrementally improves overtime.
The architecture of Kitsune is illustrated in the figure below:
- First, a feature extraction framework called AfterImage efficiently tracks the patterns of every network channel using damped incremental statisitcs, and extracts a feature vector for each packet. The vector captures the temporal context of the packet's channel and sender.
- Next, the features are mapped to the visible neurons of an ensemble of autoenoders (KitNET https://github.com/ymirsky/KitNET-py).
- Then, each autoencoder attempts to reconstruct the instance's features, and computes the reconstruction error in terms of root mean squared errors (RMSE).
- Finally, the RMSEs are forwarded to an output autoencoder, which acts as a non-linear voting mechanism for the ensemble.

## Some points about KitNET:

- It is completely plug-and-play.
- It is based on an unsupervised machine learning algorithm (it does not need label, just train it on normal data!)
- Its efficiency can be scaled with its input parameter m: the maximal size of any autoencoder in the ensemble layer (smaller autoencoders are exponentially cheaper to train and execute)

## Modifications
The Original script was modified To output CSV's instead of TSV's
## Requirements

Kitsune uses a number of open source libraries to work properly:
- time
- pandas
- numpy
- sklearn 
- scipy
- matplotlib
- dtreeviz 

## Description of the dataset
A preprocessed dataset in csv format (ready for machine learning)
Label vector corresponding to csv format Original network capture in pcap format
Characteristics of the kitsune dataset

| Attacks type | Attacks Name | Attributes | Size |                     
| ------ | ------ | ------ | ------ |
| Botnetn Malware | Mirai | 115 | 764,136 |                                        
| Denial of service | SSL Renegotiation | 115 | 2,207,570 |  
| Denial of service | SSDP Flood | 115 | 4,077,265 |  
| Denial of service | SYN DoS | 115 | 2,771,275 |  
| Man in the Middle | ARP MitM | 115 | 2,504,266 |  
| Man in the Middle | Video Injection | 115 | 2,472,400 |  
| Man in the Middle | Active Wiretap | 115 | 2,278,688 |  
| Recon | OS SCAN | 115 | 1,697,850 |  
| Recon | Fuzzing | 115 | 2,244,138 |  

## Confusion matrix

| accuracy | accuracy | precision | recall | FNR | FPR |                    
| ------ | ------ | ------ | ------ | ------ | ------ |
| Decision tree | 64.97% | 60.94% | 95.23% | 4.76% | 24.87% |      
| Random Forest | 68.36% | 64.09% | 96.84% | 3,84% | 22.07% |  
| Extra Tree | 70.05% | 68.19% | 97.92% | 3.14% | 21.07% |  
| Kitnet | 70.5% | 68.2% | 94.6% | 3.24% | 20.92% |  


#### Interpretation and Results Using Dtreeviz
<img src="https://github.com/maxamin/alternative-kitsune-version-no-autodencoders-are-used/blob/f107e4036432e45f7cf28347a5d693657de30676/Screenshots/Dtreeviz1.png"/>
<img src="https://github.com/maxamin/alternative-kitsune-version-no-autodencoders-are-used/blob/f107e4036432e45f7cf28347a5d693657de30676/Screenshots/Dtreeviz2.png"/>
<img src="https://github.com/maxamin/alternative-kitsune-version-no-autodencoders-are-used/blob/f107e4036432e45f7cf28347a5d693657de30676/Screenshots/Dtreeviz3.png"/>
#### Interpretation and Results Using Matploitlib
<img src="https://github.com/maxamin/alternative-kitsune-version-no-autodencoders-are-used/blob/f107e4036432e45f7cf28347a5d693657de30676/Screenshots/mp1.png"/>
<img src="https://github.com/maxamin/alternative-kitsune-version-no-autodencoders-are-used/blob/f107e4036432e45f7cf28347a5d693657de30676/Screenshots/mp2.png"/>
<img src="https://github.com/maxamin/alternative-kitsune-version-no-autodencoders-are-used/blob/f107e4036432e45f7cf28347a5d693657de30676/Screenshots/mp3.png"/>
