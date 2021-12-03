from Kitsune import *


# KitNET params:
maxAE = 10 #maximum size for any autoencoder in the ensemble layer
FMgrace = 500 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = 5000 #the number of instances used to train the anomaly detector (ensemble itself)
packet_limit = np.Inf #the number of packets from the input file to process
path = "Fuzzing_pcap.pcapng" #the pcap, pcapng, or tsv file which you wish to process.

# Build Kitsune
K = Kitsune(path,packet_limit,maxAE,FMgrace,ADgrace)
while True: 
    rmse = K.proc_next_packet() #will train during the grace periods, then execute on all the rest.
    K.train()
    if rmse == -1:
        break
    print(rmse)

