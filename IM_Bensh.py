from __future__ import division
import networkx as nx
import numpy as np
import networkx 
import time


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




print("data readed... !\n")


G=nx.read_edgelist(Data, nodetype = int)



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



def celf(g,k,p,mc=100):  
    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """
      
    # --------------------
    # Find the first node with greedy algorithm
    # --------------------
    
    # Calculate the first iteration sorted list
    start_time = time.time() 
    marg_gain = [IC(g,[node],p,mc) for node in range(len(g.nodes()))]

    # Create the sorted list of nodes and their marginal gain 
    Q = sorted(zip(range(len(g.nodes())),marg_gain), key=lambda x: x[1],reverse=True)

    # Select the first node and remove from candidate list
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS, timelapse = Q[1:], [len(g.nodes())], [time.time()-start_time]
    
    # --------------------
    # Find the next k-1 nodes using the list-sorting procedure
    # --------------------
    
    for _ in range(k-1):    

        check, node_lookup = False, 0
        
        while not check:
            
            # Count the number of times the spread is computed
            node_lookup += 1
            
            # Recalculate spread of top node
            current = Q[0][0]
            
            # Evaluate the spread function and store the marginal gain in the list
            Q[0] = (current,IC(g,S+[current],p,mc) - spread)

            # Re-sort the list
            Q = sorted(Q, key = lambda x: x[1], reverse = True)

            # Check if previous top node stayed on top after the sort
            check = (Q[0][0] == current)

        # Select the next node
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)
        LOOKUPS.append(node_lookup)
        timelapse.append(time.time() - start_time)

        # Remove the selected node from the list
        Q = Q[1:]

    return(S,spread,sum(timelapse)/len(timelapse))

def greedy(g,k,p,mc=100):
    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """

    S, spread, timelapse, start_time = [], [], [], time.time()
    
    # Find k nodes with largest marginal gain
    for _ in range(k):

        # Loop over nodes that are not yet in seed set to find biggest marginal gain
        best_spread = 0
        for j in set(range(len(list(g.nodes()))))-set(S):

            # Get the spread
            s = IC(g,S + [j],p,mc)

            # Update the winning node and spread so far
            if s > best_spread:
                best_spread, node = s, j

        # Add the selected node to the seed set
        S.append(node)
        
        # Add estimated spread and elapsed time
        spread.append(best_spread)
        timelapse.append(time.time() - start_time)

    return(S,round(sum(spread)/len(spread)),sum(timelapse)/len(timelapse))




k=4
p=0.1
print("k = ", k , "\np = ",p)
#CELF approach
S=celf(G,k,p)

#Greedy approach
#e=greedy(G,k,p)

print("Starting with CELF approach : \n")
print("the seed set of CELF is :", S[0]," max spread = ",S[1],"  and runnung time was :  ",S[2])

#print("\nMoving to GREEDY :\n ")   
#print("the seed set of GREEDY is :", e[0]," max spread = ",e[1],"  and runnung time was :  ",e[2])    
