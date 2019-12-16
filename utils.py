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
