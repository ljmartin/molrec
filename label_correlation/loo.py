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

#this function removes the influence of one ligand from the correlation graph, i.e. LOO.
def subtractLigand(L, row, tot_instances):
    P = copy.copy(L) #don't want to change the values in L or it will change over time. 
                    #take a copy each time we want to subtract a ligand, to keep L constant.
    for j,k in itertools.permutations(np.where(row==1)[0], 2):
        P[j][k] -= (1)/(tot_instances[k])  
    return 1-P


tot_instances = np.sum(y, axis=0)
trials=1000
rank_list = list()
hit_list = list()
miss_list = list()

##Do LOO ranking:
for idx in tqdm(multiple_labels.nonzero()[0]): #for every multilabel ligand
    labels = y[idx] #fetch this ligand's labels
    L1 = subtractLigand(L, y[idx], tot_instances) #remove knowledge of this ligand from the correlation matrix:
    ones = np.where(labels==1)[0] #positive label indices
    zeros = np.where(labels==0)[0] #unknown label indices
    
    scores = np.zeros(labels.shape[0])
    for o in ones: #for each positive, set it's label to zero. Then calculate probability of a hit 
        labels = copy.copy(y[idx])
        labels[o]=0
        probs_pos = 1-np.prod(L1[o][labels==1])
        scores[o]=probs_pos
    for z in zeros: #for each zero, calculate the probability of a hit
        labels = copy.copy(y[idx])
        probs_pos = 1-np.prod(L1[z][labels==1])
        scores[z]=probs_pos

    #these are the ranks of the predicted probabilities for this ligand
    ranks_unadjusted = np.argsort(-scores).argsort()[labels.astype(bool)]+1
    
    #now increase the worse ranked positives by the number of positives in front of it
    #for example if two ligands ranked 1, 2, we count it as 1,1
    #this is because we don't want to penalise perfect ranking of two ligands in a row. 
    ranks_adj = ranks_unadjusted-(ranks_unadjusted).argsort().argsort()

    #for calculating calibration:
    hit_list += list(scores[labels.astype(bool)])
    miss_list += list(scores[~labels.astype(bool)])

    rank_list += list(ranks_adj)


rank_arr = np.array(rank_list)
np.save('rank_arr.npy', rank_arr)
np.save('hit_arr.npy', np.array(hit_list))
np.save('miss_arr.npy', np.array(miss_list))
