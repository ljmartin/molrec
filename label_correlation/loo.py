from scipy import sparse
import numpy as np
import itertools
import copy

from tqdm import tqdm

import sys
sys.path.append("..")
import utils

#all labels:
interaction_matrix = sparse.load_npz('../data/interaction_matrix_pchembl.npz')
y = interaction_matrix.toarray()
#correlation matrix:
L = utils.makeCorrelations(interaction_matrix).toarray()

# get ligands with more than one label.
# Because you can't make predictions from the correlation matrix 
# when you've removed the only known positive.
multiple_labels = np.sum(y,axis=1)>1

#morgan fingerprints for similarity calculations:
fps = sparse.load_npz('../time_split/morgan.npz')
fps_multiples = fps[multiple_labels]
y_multiples = y[multiple_labels]

#####
##
## Step 0: we have to calculate a lot of Dice distances here, 
## so here's a Dice distance calculator that is faster than
## scipy's because it takes advantage of sparsity:
##
#####
def fast_dice(X, Y=None):
    if isinstance(X, np.ndarray):
        X = sparse.csr_matrix(X).astype(bool).astype(int)
    if Y is None:
        Y = X
    else:
        if isinstance(Y, np.ndarray):
            Y = sparse.csr_matrix(Y).astype(bool).astype(int)

    intersect = X.dot(Y.T)
    #cardinality = X.sum(1).A
    cardinality_X = X.getnnz(1)[:,None] #slightly faster on large matrices - 13s vs 16s for 12k x 12k
    cardinality_Y = Y.getnnz(1) #slightly faster on large matrices - 13s vs 16s for 12k x 12k
    return (1-(2*intersect) / (cardinality_X+cardinality_Y.T)).A




#####
##
##Step 1: calculate predicted ranking for singly held-out ligands, i.e. leave-one-out cross-validation.
##
#####

#this function removes the influence of one ligand from the correlation graph, i.e. LOO.
def subtractLigand(L, row, tot_instances):
    P = copy.copy(L) #don't want to change the values in L or it will change over time. 
                    #take a copy each time we want to subtract a ligand, to keep L constant.
    for j,k in itertools.permutations(np.where(row==1)[0], 2):
        P[j][k] -= (1)/(tot_instances[k])  
    return 1-P



num_trials = 5000
sample = np.random.choice(multiple_labels.nonzero()[0], num_trials)

tot_instances = np.sum(y, axis=0)
rank_list = list()
hit_list = list()
miss_list = list()
similarities = list()

##Do LOO ranking:
for idx in tqdm(sample): #for every multilabel ligand
    labels = y[idx] #fetch this ligand's labels
    L1 = subtractLigand(L, y[idx], tot_instances) #remove knowledge of this ligand from the correlation matrix:
    ones = np.where(labels==1)[0] #positive label indices
    zeros = np.where(labels==0)[0] #unknown label indices

    scores = np.zeros(labels.shape[0])
    
    #For each positive label, set it's label to zero ('leaving it out'). 
    #Then calculate the probability it is a hit
    for o in ones:
        labels = copy.copy(y[idx])
        labels[o]=0
        probs_pos = 1-np.prod(L1[o][labels==1])
        scores[o]=probs_pos
    
    #Do the same for the negatives, i.e. for each zero label, calculate the probability of a hit
    for z in zeros: 
        labels = copy.copy(y[idx])
        probs_pos = 1-np.prod(L1[z][labels==1])
        scores[z]=probs_pos
        
    #now, separately to ranking, also calculate 
    #the Morgan fingerprint distances to ALL other positive labels. 
    for o in ones:
        test_ligand = fps[idx]
        other_ligand_indices = np.where(y[:,o]==1)[0] #all other positives with the same label.
        other_ligands = fps[other_ligand_indices] #this is their fingerprints
        nnsimilarities =fast_dice(test_ligand, other_ligands)[0]
        
        similarities.append(nnsimilarities)


    #Finally, calculate the ranks of the held out ligand vs other poitives and the negatives:

    #these are the rankings of the predicted probabilities for this ligand
    ranks_unadjusted = np.argsort(-scores).argsort()[labels.astype(bool)]+1

    #must adjust the rankings to handle cases where two positive ligands are ranked in a row. 
    #for example if two ligands ranked 1, 2, we must record it as 1,1
    #this is to avoid penalising perfect ranking of two ligands in a row.
    ranks_adj = ranks_unadjusted-(ranks_unadjusted).argsort().argsort()

    #now record the probabilities for the positives and negatives, 
    #which will be used for for calculating calibration:
    hit_list += list(scores[labels.astype(bool)])
    miss_list += list(scores[~labels.astype(bool)])

    rank_list += list(ranks_adj)

np.save('rank_arr_full_data.npy', np.array(rank_list))
np.save('hit_arr.npy', np.array(hit_list))
np.save('miss_arr.npy', np.array(miss_list))

#we will take the mean similarity to the top 30 most similar ligands:
#we use 1:31 because the most-similar ligand is self similarity (value of 0)
N=31
top_30_mean = np.array([i[np.argsort(i)][1:N].mean() for i in similarities])
np.save('nn_distances_full_data.npy', top_30_mean)


#####
##
## Step 2: same as above, but while ALSO holding out nearest neighbors. 
## The scoring isn't performed on the nearest neighbors, just on the single hold-out,
## but we remove the influence of the number of ligands that are nearest neighbors,
## defined by some distance cutoff. 
## 
#####


#this function removes the influence of MULTIPLE ligands from the correlation graph
def subtractMultipleLigands(L, rows, tot_instances):
    P = copy.copy(L) #don't want to change the values in L or it will change over time.
                    #take a copy each time we want to subtract a ligand, to keep L constant.
    for row in rows:
        if row.sum()>1:
            for j,k in itertools.permutations(np.where(row==1)[0], 2):
                P[j][k] -= (1)/(tot_instances[k])
    return 1-P

sample = np.random.choice(multiple_labels.nonzero()[0], num_trials)

tot_instances = np.sum(y, axis=0)
nn_removed_rank_list = list()
nn_removed_similarities = list()

#distance cutoff used for removing nearest neighbors. 
nn_cutoff = 0.4

##Do LOO ranking:
for idx in tqdm(sample): #for every multilabel ligand
    labels = y[idx] #fetch this ligand's labels
    
    ones = np.where(labels==1)[0] #positive label indices
    zeros = np.where(labels==0)[0] #unknown label indices

    scores = np.zeros(labels.shape[0])
    
    #For each positive label, set it's label to zero ('leaving it out'). 
    #Then calculate the probability it would be a hit
    ##This time, however, also remove labels of all nearest neighbors!
    for o in ones: 
        labels = copy.copy(y[idx])
        labels[o]=0 #set the left-out ligand label to 0. 
        
        #remove the labels from the nearest neighbours (i.e. with distance<cutoff):
        test_ligand = fps[idx] #this is the left-out ligand
        other_ligand_indices = np.where(y[:,o]==1)[0] #all other positives with the same label.
        other_ligands = fps[other_ligand_indices] #this is their fingerprints
        nnsimilarities =fast_dice(test_ligand, other_ligands)[0] #list of Dice distances
        
        nearest_neighbor_mask = nnsimilarities<nn_cutoff #we will remove all of these nearest neighbors
        nearest_neighbor_indices = other_ligand_indices[nearest_neighbor_mask]
        #remove knowledge of this ligand from the correlation matrix:
        L_multiple_ligands_removed = subtractMultipleLigands(L, y[nearest_neighbor_indices], tot_instances) 

        #Now, like the first time, calculate the probability this left-out ligand is a hit:
        probs_pos = 1-np.prod(L_multiple_ligands_removed[o][labels==1])
        scores[o]=probs_pos
        
        #since we have already calculated nearest-neighbor distances, 
        #might as well save that list now instead of calculating it afterwards
        remaining_nnsimilarities = nnsimilarities[~nearest_neighbor_mask]         
        nn_removed_similarities.append(remaining_nnsimilarities)
        
    for z in zeros: #for each zero, calculate the probability of a hit
        labels = copy.copy(y[idx])
        probs_pos = 1-np.prod(L1[z][labels==1])
        scores[z]=probs_pos

    #Finally, calculate the ranks of the held out ligand vs other poitives and the negatives:

    #these are the ranks of the predicted probabilities for this ligand
    ranks_unadjusted = np.argsort(-scores).argsort()[labels.astype(bool)]+1

    #must adjust the rankings to handle cases where two positive ligands are ranked in a row.
    #for example if two ligands ranked 1, 2, we must record it as 1,1
    #this is to avoid penalising perfect ranking of two ligands in a row.
    ranks_adj = ranks_unadjusted-(ranks_unadjusted).argsort().argsort()

    nn_removed_rank_list += list(ranks_adj)
    


np.save('rank_arr_nn_removed.npy', np.array(nn_removed_rank_list))
#we will take the mean similarity to the top 30 most similar ligands:
#we use 0:30 because the most-similar ligand has been removed
N=30
top_30_mean = np.array([i[np.argsort(i)][1:N].mean() for i in nn_removed_similarities])
np.save('nn_distances_nn_removed.npy', top_30_mean)

