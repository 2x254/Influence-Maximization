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



def transfrom_list_of_sets_to_list_of_lists(list_comm):
    biglist=[]
    for e in list_comm:
        sublist=list(e)
        biglist.append(sublist)
    return biglist

def get_communities_from_txt(path):
    file1 = open(path, 'r')
    Lines = file1.readlines()
    comms=[]
    for line in Lines:
        comms.append(line.split())
    return comms

def tansform_comm_labsl_str(comm):
    l1=[]
    for e in comm:
        subl1=[]
        for el in e:
            subl1.append(str(el))
        l1.append(subl1)
    return l1
        
def tansform_comm_labsl_int(comm):
    l1=[]
    for e in comm:
        subl1=[]
        for el in e:
            subl1.append(int(el))
        l1.append(subl1)
    return l1            
        
    
    
# synthetic networks LFR_benchmark :   min_comm=50 average_density=5 taux1=3 taux2=1.5
# 1000 nodes and mu between 0.1 and 0.9
#Data = open( 'datasets/LFR/LFR_N1000_ad5_mc50_mu0.1.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N1000_ad5_mc50_mu0.1.json").communities
#Data = open( 'datasets/LFR/LFR_N1000_ad5_mc50_mu0.2.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N1000_ad5_mc50_mu0.2.json").communities
#Data = open( 'datasets/LFR/LFR_N1000_ad5_mc50_mu0.3.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N1000_ad5_mc50_mu0.3.json").communities
#Data = open( 'datasets/LFR/LFR_N1000_ad5_mc50_mu0.4.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N1000_ad5_mc50_mu0.4.json").communities
#Data = open( 'datasets/LFR/LFR_N1000_ad5_mc50_mu0.5.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N1000_ad5_mc50_mu0.5.json").communities
#Data = open( 'datasets/LFR/LFR_N1000_ad5_mc50_mu0.6.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N1000_ad5_mc50_mu0.6.json").communities
#Data = open( 'datasets/LFR/LFR_N1000_ad5_mc50_mu0.7.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N1000_ad5_mc50_mu0.7.json").communities
#Data = open( 'datasets/LFR/LFR_N1000_ad5_mc50_mu0.8.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N1000_ad5_mc50_mu0.8.json").communities
#Data = open( 'datasets/LFR/LFR_N1000_ad5_mc50_mu0.9.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N1000_ad5_mc50_mu0.9.json").communities

# 5000 nodes and mu between 0.1 and 0.9
#Data = open( 'datasets/LFR/LFR_N5000_ad5_mc50_mu0.1.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N5000_ad5_mc50_mu0.1.json").communities
#Data = open( 'datasets/LFR/LFR_N5000_ad5_mc50_mu0.2.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N5000_ad5_mc50_mu0.2.json").communities
#Data = open( 'datasets/LFR/LFR_N5000_ad5_mc50_mu0.3.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N5000_ad5_mc50_mu0.3.json").communities
#Data = open( 'datasets/LFR/LFR_N5000_ad5_mc50_mu0.4.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N5000_ad5_mc50_mu0.4.json").communities
#Data = open( 'datasets/LFR/LFR_N5000_ad5_mc50_mu0.5.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N5000_ad5_mc50_mu0.5.json").communities
#Data = open( 'datasets/LFR/LFR_N5000_ad5_mc50_mu0.6.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N5000_ad5_mc50_mu0.6.json").communities
#Data = open( 'datasets/LFR/LFR_N5000_ad5_mc50_mu0.7.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N5000_ad5_mc50_mu0.7.json").communities
#Data = open( 'datasets/LFR/LFR_N5000_ad5_mc50_mu0.8.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N5000_ad5_mc50_mu0.8.json").communities
#Data = open( 'datasets/LFR/LFR_N5000_ad5_mc50_mu0.9.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N5000_ad5_mc50_mu0.9.json").communities


# 10000 nodes and mu between 0.1 and 0.9
#Data = open( 'datasets/LFR/LFR_N10000_ad5_mc50_mu0.1.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N10000_ad5_mc50_mu0.1.json").communities
#Data = open( 'datasets/LFR/LFR_N10000_ad5_mc50_mu0.2.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N10000_ad5_mc50_mu0.2.json").communities
#Data = open( 'datasets/LFR/LFR_N10000_ad5_mc50_mu0.3.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N10000_ad5_mc50_mu0.3.json").communities
#Data = open( 'datasets/LFR/LFR_N10000_ad5_mc50_mu0.4.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N10000_ad5_mc50_mu0.4.json").communities
#Data = open( 'datasets/LFR/LFR_N10000_ad5_mc50_mu0.5.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N10000_ad5_mc50_mu0.5.json").communities
#Data = open( 'datasets/LFR/LFR_N10000_ad5_mc50_mu0.6.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N10000_ad5_mc50_mu0.6.json").communities
#Data = open( 'datasets/LFR/LFR_N10000_ad5_mc50_mu0.7.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N10000_ad5_mc50_mu0.7.json").communities
#Data = open( 'datasets/LFR/LFR_N10000_ad5_mc50_mu0.8.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N10000_ad5_mc50_mu0.8.json").communities
#Data = open( 'datasets/LFR/LFR_N10000_ad5_mc50_mu0.9.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N10000_ad5_mc50_mu0.9.json").communities

# 50000 nodes and mu between 0.1 and 0.9
#Data = open( 'datasets/LFR/LFR_N50000_ad5_mc50_mu0.1.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N50000_ad5_mc50_mu0.1.json").communities
#Data = open( 'datasets/LFR/LFR_N50000_ad5_mc50_mu0.2.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N50000_ad5_mc50_mu0.2.json").communities
#Data = open( 'datasets/LFR/LFR_N50000_ad5_mc50_mu0.3.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N50000_ad5_mc50_mu0.3.json").communities
#Data = open( 'datasets/LFR/LFR_N50000_ad5_mc50_mu0.4.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N50000_ad5_mc50_mu0.4.json").communities
#Data = open( 'datasets/LFR/LFR_N50000_ad5_mc50_mu0.5.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N50000_ad5_mc50_mu0.5.json").communities
#Data = open( 'datasets/LFR/LFR_N50000_ad5_mc50_mu0.6.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N50000_ad5_mc50_mu0.6.json").communities
#Data = open( 'datasets/LFR/LFR_N50000_ad5_mc50_mu0.7.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N50000_ad5_mc50_mu0.7.json").communities
#Data = open( 'datasets/LFR/LFR_N50000_ad5_mc50_mu0.8.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N50000_ad5_mc50_mu0.8.json").communities
#Data = open( 'datasets/LFR/LFR_N50000_ad5_mc50_mu0.9.csv', "r")
#communities=readwrite.read_community_json( "datasets/LFR/LFR_N50000_ad5_mc50_mu0.9.json").communities


print("getting data...")
#real data sets 
#Data=open( "datasets/amazon/Amazon.csv","r")
#communities=readwrite.read_community_json( "datasets/amazon/amazon_Truth.json").communities

#Data = open( "datasets/karate/karate.csv","r")
#communities=readwrite.read_community_json( "datasets/karate/karate_Truth.json").communities

#Data = open( "datasets/emailEuCore/emailEucore.csv","r")
#communities=readwrite.read_community_json( "datasets/emailEuCore/email_Truth.json").communities

Data = open( "datasets/dolphins/dolphins.csv","r")
communities=readwrite.read_community_json( "datasets/dolphins/dolphins_truth.json").communities

#Data = open( "datasets/Books/Books.csv","r")
#communities=readwrite.read_community_json( "datasets/Books/Books_truth.json").communities

#Data = open("datasets/cora/cora.csv","r")
#communities=readwrite.read_community_json( "datasets/cora/cora_truth.json").communities

#Data = open("datasets/citeseer/citeseer.csv","r")
#communities=readwrite.read_community_json("datasets/citeseer/citeseer_truth.json").communities

#Data = open("datasets/youtube/youtube.csv","r")
#communities=readwrite.read_community_json("datasets/youtube/youtube.json").communities


G = nx.parse_edgelist(Data,nodetype=int)

#(1)  calculate similarity
print("calculating similarities")
for (u,v) in G.edges():
    adjU=set(list(G.neighbors(u))).union({u})
    adjV=set(list(G.neighbors(v))).union({v})
    G[u][v]['weight']=len(adjU.intersection(adjV))/math.sqrt(len(adjU)*len(adjV))
    



def Getlist_simlarity(Graph):
    sim_list=[]
    for (u,v,d) in Graph.edges(data=True):
        sim_list.append(d['weight'])
    return list(dict.fromkeys(sim_list))

def Get_Max_Edges_sets(Graph,maxi_weight):
    maxi_edges=[]
    for (u,v,d) in Graph.edges(data=True):
        if d['weight'] == maxi_weight:
            maxi_edges.append((u,v))
    return maxi_edges


def to_edges(l):
    it = iter(l)
    last = next(it)
    for current in it:
        yield last, current
        last = current 
def to_graph(l):
    gr = networkx.Graph()
    for part in l:
        gr.add_nodes_from(part)
        gr.add_edges_from(to_edges(part))
    return gr

def Merge_edges(List_edges):
    gra = to_graph(List_edges)
    return list(connected_components(gra))
    

def init_commnuities(Graph):
    Com=[]
    for node in Graph.nodes():
        Com.append({node})
    return Com


#(2) detect communities  
print("detecting comms....")
current_clustering_list=init_commnuities(G)
current_modularity=modularity(G,current_clustering_list,weight='weight')
list_weights=Getlist_simlarity(G)

while True:
   
    if list_weights:
        maxi_weight=max(list_weights)
    else:
        break
    maxi_edges_sets=Get_Max_Edges_sets(G,maxi_weight)
    maxi_edges_sets_merged=list(nx1.connected_components(nx1.Graph(maxi_edges_sets)))
    previous_clustering_list=current_clustering_list
    previous_modularity=current_modularity
    current_clustering_list=Merge_edges(current_clustering_list+maxi_edges_sets_merged)
    current_modularity=modularity(G,current_clustering_list,weight='weight')
    if current_modularity<previous_modularity:
        break
    list_weights.remove(maxi_weight)
if current_modularity>=previous_modularity:
    result=current_clustering_list
   
else:
    result=previous_clustering_list

#(3) evaluating community quality according to NMI and F1-score




communities = [set(p) for p in communities]

res_omega_nmi=NodeClustering(result, graph=G, method_name="normalized_mutual_information")

communities_omega_nmi=NodeClustering(communities, graph=G, method_name="normalized_mutual_information")


print("f1-score....")
#f1 score
f1=evaluation.f1(communities_omega_nmi,res_omega_nmi)
print("\n\nF1 score = ",f1.score)


#Nmi
print("NMI......")
nmi=evaluation.normalized_mutual_information(communities_omega_nmi,res_omega_nmi)
print("\n\nNMI = ",nmi.score)



