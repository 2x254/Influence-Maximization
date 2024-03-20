from __future__ import division
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt 
import math 
from scipy.io import mmread
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.community.quality import performance
import random
from distinctipy import distinctipy
from networkx.algorithms import community
from cdlib import evaluation,algorithms,NodeClustering
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from cdlib import readwrite
from cdlib import readwrite
import numpy as np
import csv
import networkx as nx1
import networkx 
from networkx.algorithms.components.connected import connected_components
from igraph import clustering  
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import markov_clustering as mc


#real data sets 
#Data=open( "datasets/amazon/Amazon.csv","r")
#communities=readwrite.read_community_json( "datasets/amazon/amazon_Truth.json").communities

#Data = open( "datasets/karate/karate.csv","r")
#communities=readwrite.read_community_json( "datasets/karate/karate_Truth.json").communities

#Data = open( "datasets/emailEuCore/emailEucore.csv","r")
#communities=readwrite.read_community_json( "datasets/emailEuCore/email_Truth.json").communities

#Data = open( "datasets/dolphins/dolphins.csv","r")
#communities=readwrite.read_community_json( "datasets/dolphins/dolphins_truth.json").communities

Data = open( "datasets/Books/Books.csv","r")
communities=readwrite.read_community_json( "datasets/Books/Books_truth.json").communities

#Data = open("datasets/cora/cora.csv","r")
#communities=readwrite.read_community_json( "datasets/cora/cora_truth.json").communities

#Data = open("datasets/citeseer/citeseer.csv","r")
#communities=readwrite.read_community_json("datasets/citeseer/citeseer_truth.json").communities

#Data = open("datasets/youtube/youtube.csv","r")
#communities=readwrite.read_community_json("datasets/youtube/youtube.json").communities

G = nx.parse_edgelist(Data,nodetype=int)

#print("detecting...... chosen approach....\n")
#res_omega_nmi= algorithms.label_propagation(G)
res_omega_nmi=algorithms.leiden(G)
#res_omega_nmi=algorithms.surprise_communities(G)
#res_omega_nmi=algorithms.significance_communities(G)

#for the MC approach
"""
A = nx.to_numpy_matrix(G,dtype=np.uint8)
result = mc.run_mcl(A)
clusters = mc.get_clusters(result)
result=[list(p) for p in clusters]
res_omega_nmi=NodeClustering(result, graph=G, method_name="normalized_mutual_information")
"""


communities_omega_nmi=NodeClustering(communities, graph=G, method_name="normalized_mutual_information")




#NMI

nmi=evaluation.normalized_mutual_information(res_omega_nmi,communities_omega_nmi)
print("\n\nNMI = ",nmi.score)



#F1-score

print("f1 score.....:\n")
#f1 score
f1=evaluation.f1(communities_omega_nmi,res_omega_nmi)
print("\n\nF1 score = ",f1.score)








