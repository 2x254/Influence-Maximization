
from __future__ import division
import networkx as nx
import sys
import math 

from networkx.algorithms.community.quality import modularity

import numpy as np

import networkx as nx1
import networkx 
import time

from networkx.algorithms.components.connected import connected_components





def transfrom_list_of_sets_to_list_of_lists(list_comm):
    biglist=[list(e) for e in list_comm]
    return biglist

def get_communities_from_txt(path):
    file1 = open(path, 'r')
    Lines = file1.readlines()
    comms=[line.split() for line in Lines]
    file1.close()
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
        
    
print("reading data....") 



print("getting data...")
#real data sets 


Data = open( "datasets/karate/karate.csv","r")
print("\nKarate data ....\n")

#Data = open( "datasets/dolphins/dolphins.csv","r")
#print("\ndolphins data ....\n")

#Data = open("datasets/Books/Books.csv","r")
#print("\nbooks data ...")


#Data = open("datasets/facebook/Facebook_artist.csv","r")
#print("\nfacebook data artist....\n")

#Data = open("datasets/emailEuCore/emailEucore.csv","r")
#print("\nemail data ...")






print("data readed... !")


G=nx.read_edgelist(Data, nodetype = int)


# functions

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




# (1) calculate similarity
#
#parameters
k=10
p=0.1

print("k = ", k , "\np = ",p)
alpha=1


print("\nwith alpha = ",alpha,"\n")
print("\ncalculating similarities...\n")
start=time.time()
for (u,v) in G.edges():
    adjU=set(list(G.neighbors(u))).union({u})
    adjV=set(list(G.neighbors(v))).union({v})
    G[u][v]['weight']=((1-alpha)*(len(adjU.intersection(adjV))/math.sqrt(len(adjU)*len(adjV))))+((alpha)*(len(G.subgraph(adjU.intersection(adjV)).edges())/min(len(G.subgraph(adjU).edges()),len(G.subgraph(adjV).edges()))))
    #G[u][v]['weight']=len(adjU.intersection(adjV))/math.sqrt(len(adjU)*len(adjV))
    

     
#print("similarities calculated ! \n")
 


#(2) detect communities  

print("detecting comms....\n")
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

#Searching for  seed nodes: 
    
print("\nSearching for seed nodes ..... \n")

#(3) extract most influencual nodes 

result=[list(c) for c in result]
#print(result)
communities_ones=[]
communtites_lots=[]
for c in result:
    if len(c)==1:
        communities_ones.append(c)
    else:
        communtites_lots.append(c)


sorted_communtites_lots=sorted(communtites_lots, key=len, reverse=True)


#define  the propagation models
def IC(G,S,p,mc=100):
   
    spread = []
    for i in range(mc):
        
         
        new_active, A = S[:], S[:]
        while new_active:

          
            new_ones = []
            for node in new_active:
                
              
                np.random.seed(i)
                success = np.random.uniform(0,1,len(list(G.neighbors(node)))) < p
                new_ones += list(np.extract(success, list(G.neighbors(node))))

            new_active = list(set(new_ones) - set(A))
            
         
            A += new_active
            
        spread.append(len(A))
        
    return(round(np.mean(spread)))

def getscoreNode(node, G):
    voisinnode=set(list(G.neighbors(node)))
    set_golobal_nodes=set()
    for v in voisinnode:
        set_golobal_nodes.union(set(list(G.neighbors(v))))
    H=G.subgraph(set_golobal_nodes)
    score =0
    for (u,v) in H.edges():
        Nu=set(list(G.neighbors(u)))
        Nv=set(list(G.neighbors(v)))
        score= score +len(Nu.intersection(Nv))
    return score


scoresno=[]
for no in communities_ones:
    scoresno.append((no,getscoreNode(no[0],G)))
scoresno=sorted(scoresno, key=lambda x: x[1])


def get_best_spreader_node(Graph):
    list_no_spread=[]
    for nod in Graph:
        list_no_spread.append((nod,IC(G,[nod],p)))
    best_nod=max(list_no_spread, key=lambda x: x[1])
    return best_nod[0]

def number_nodes_in_coms(list_of_lists):
    total= 0
    for sublist in list_of_lists:
        total=total+len(sublist)
    return total

# 
def get_seed_set(k,G,sorted_communtites_lots):
    seed_set=[] 
    while len(seed_set)<k:
        for c in sorted_communtites_lots:
            if len(c)>0:
                G_c=G.subgraph(c)
                best_node = get_best_spreader_node(G_c)
                if len(seed_set)<k:
                    
                    seed_set.append(best_node)
                    
                    c.remove(best_node)
                else:
                    break
    return seed_set
#k=10
#print("`\nk = ",k,"\n")
seed_set=[]
if k > len(list(G.nodes())):
    exit_msg = "k should not be higher than the number of nodes."
    sys.exit(exit_msg)

elif k > (len(list(G.nodes())) - len(communities_ones)):
    seed_set=get_seed_set(len(list(G.nodes())) - len(communities_ones),G,sorted_communtites_lots)
    
    for el in scoresno:
        if len(seed_set)<k:
            seed_set.append(el[0][0])
        else:
            break
else:
    seed_set=get_seed_set(k,G,sorted_communtites_lots)



#enhancement for selecting seed nodes

spreadingone=IC(G, seed_set,p)
for ie in scoresno:
    rep=seed_set[len(seed_set)-1]
    seed_set.remove(rep)
    seed_set.append(ie[0][0])
    spreadingtwo=IC(G, seed_set,p)
    if spreadingtwo>spreadingone:
        break
    seed_set.remove(ie[0][0])
    seed_set.append(rep)
 
       
print(seed_set," having the spreading = ", IC(G, seed_set,p),"with running time : ",time.time() - start )









