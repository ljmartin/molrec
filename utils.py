import numpy as np
from scipy import sparse
import copy

def train_test_split(input_matrix, fraction):
    """
    Splits a label matrix ("y" in the sklearn style), with rows being 
    instances and columns being labels, into test and train matrices. 

    Because the intended use is network-based algorithms, the train
    matrix has to have some examples for each instance - 
    so we can't just remove a fraction of the instances like in normal cross validation. 

    Instead, this iterates through the labels and removes a percentage of them
    into the test matrix. If all previous labels have been removed, the final label
    is skipped. This might lead to bias in which labels get removed, so 
    the columns are randomly shuffled first. 

    :param input_matrix: 2D numpy array of size (n,m) for dataset with
    n instances and m labls. Must be 1 for label, and 0 for absence of a label.
    :param fraction: percentage size of the test matrix. 


    """

    indices = np.arange(input_matrix.shape[1]) 
    np.random.shuffle(indices)
    y = input_matrix[:,indices]
    
    train = copy.copy(y)
    test = np.zeros([y.shape[0], y.shape[1]])
    
    for count, row in enumerate(train):
        ones = row.nonzero()[0]
        numligs = len(ones) 
        for o in ones:
            if np.random.uniform(0,1)<fraction:
                if numligs>1:#so we don't remove all labels from an instance
                    numligs-=1
                    train[count][o]=0
                    test[count][o]=1
                
    return sparse.csr_matrix(train), sparse.csr_matrix(test)

def evaluate_predictions(prediction_matrix, test):   
    """
    Input a numpy array, with rows for instances and columns for labels, 
    with entries containing predicted interaction scores. Usually, the higher
    the highest interaction score corresponds to the top predicted ligand,
    and thus function assumes the most positive score is the best. 

    Calculates the ranks of the test ligands and returns the mean rank. 
    This is to be optimized (i.e. minimized) by scikit-optimize. 

    :param prediction_matrix: n by m np array (n = number of instances, m = number of labels)
    containg predicted interaction scores resulting from some recommender algorithm 
    :param test: n by m sparse matrix containing 1's in the positions of each test label. Returned
    by train_test_split.
    """
    #order from highest to lowest:
    order = (-prediction_matrix).argsort()
    #get ranks of each ligand. 
    ranks = order.argsort()
    
    #calc rank fo each ligand
    test = np.array(test.todense())
    test_ranks = ranks[np.array(test, dtype=bool)]
    
    return np.mean(test_ranks)

def load_subset():
    """
    Loads the 243-target subset of data for HPO. 
    Also removes ligands with only 1 label in the dataset -
    these aren't useful since test/train splits require each 
    instance has at least one label in the training set and
    at least one label in the test set. 
    """

    interaction_matrix = sparse.load_npz('../data/interaction_matrix.npz')
    interaction_matrix = np.array(interaction_matrix.todense())

    mask = np.sum(interaction_matrix, axis=1)
    mask = mask>1

    interaction_matrix = interaction_matrix[mask] #remove instances with only 1 label
    #possible that some labels don't have ANY instances now... removing columns
    #with no labels:
    interaction_matrix = interaction_matrix[:,np.sum(interaction_matrix, axis=0)>0]

    return interaction_matrix

def load_time_split(year=2015):
    """
    Get the interaction matrix of the 243-target subset. Then
    split into two matrices, train and test, based on year of 
    the interactions. 
    :param year: split point. All interactions from this year 
    or afterwards become test interactions. 
    """

    interaction_matrix = sparse.load_npz('../data/interaction_matrix.npz')
    interaction_matrix = np.array(interaction_matrix.todense())

    interaction_dates = sparse.load_npz('../data/interaction_dates.npz')
    interaction_dates = np.array(interaction_dates.todense())

    split = np.array(interaction_dates<=year, dtype=int)

    train = interaction_matrix*split
    test = interaction_matrix - train

    return train, test
