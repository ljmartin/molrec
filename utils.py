def train_test_split(input_matrix, fraction):
    """
    Splits a label matrix ("y" in the sklearn style), with rows being 
    instances and columns being labels, into test and train matrices. 

    Because the intended use is network-based algorithms, the train
    matrix has to have some examples for each instance - 
    so we can't just remove a fraction of the instances like in normal cross validation. 

    Instead, this iterates through the labels and removes a percentage of them
    into the test matrix. Some may still have all labels removed by bad luck.
    Those will be dealt with at the evaluation stage. 

    The column are first shuffled as randomization. 

    :param input_matrix: 2D numpy array of size (n,m) for dataset with
    n instances and m labls. Must be 1 for label, and 0 for absence of a label.
    :param fraction: percentage size of the test matrix. 


    """

    indices = np.arange(input_matrix.shape[1]) 
    np.random.shuffle(indices)
    y = matrix[:,indices]
    
    train = copy.copy(y)
    test = np.zeros([y.shape[0], y.shape[1]])
    #test = copy.copy(y)
    
    for count, row in enumerate(train):
        ones = row.nonzero()[0]
        for o in ones:
            if np.random.uniform(0,1)<fraction:
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
