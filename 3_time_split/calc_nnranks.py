import tqdm
import numpy as np
from scipy import sparse

import sys
sys.path.append("..")
import utils



##The following is to calculate AVE bias:
def fast_jaccard(X, Y=None):
    """credit: https://stackoverflow.com/questions/32805916/compute-jaccard-distances-on-sparse-matrix"""
    if isinstance(X, np.ndarray):
        X = sparse.csr_matrix(X)
    if Y is None:
        Y = X
    else:
        if isinstance(Y, np.ndarray):
            Y = sparse.csr_matrix(Y)
    assert X.shape[1] == Y.shape[1]

    X = X.astype(bool).astype(int)
    Y = Y.astype(bool).astype(int)
    intersect = X.dot(Y.T)
    x_sum = X.sum(axis=1).A1
    y_sum = Y.sum(axis=1).A1
    xx, yy = np.meshgrid(x_sum, y_sum)
    union = ((xx + yy).T - intersect)
    return (1 - intersect / union).A


def get_nnrank_of_target(ligand_idx, target_idx, train_matrix):
        
    positives = train_matrix[ligand_idx].nonzero()[1]
    all_distances = fast_jaccard(fps[ligand_idx], fps)[0]
    s = np.argsort(all_distances)
    
    pred = target_idx
    curr_rank = 0
    count=1
    preds = []
    seen = []

    while pred not in seen:
    #for _ in range(100):
        predictions = train_matrix[s[count]].nonzero()[1]
    
        preds = np.setdiff1d(predictions,positives)
        preds = np.setdiff1d(preds, seen)
        
        curr_rank = len(seen)
        
        seen += list(preds)
        if len(preds)>0:
             curr_rank+= np.mean(np.arange(len(preds))+1)
        count+=1

    return curr_rank



if __name__=="__main__":
    #load time split and fingerprints:
    train, test, fps = utils.load_time_split(year=2015, return_fingerprints=True)
    #all labels:
    interaction_matrix = sparse.load_npz('../0_data/interaction_matrix_pchembl.npz')


    row_idx, col_idx = test.nonzero()
    nnranks = list()

    for lig_idx, targ_idx in tqdm.tqdm(zip(row_idx, col_idx), total=len(row_idx)):
        nnrank = get_nnrank_of_target(lig_idx, targ_idx, train)
        nnranks.append(nnrank)

    np.save('./processed_data/2015_nearest_neighbor.npy', nnranks)
