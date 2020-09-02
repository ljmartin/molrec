import numpy as np
from scipy import sparse
from scipy.stats.mstats import rankdata #for dealing with ties
import copy
import itertools
import implicit
import lightfm
from tqdm import tqdm

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

def evaluate_predictions(predictions, test, train):   
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
#    :param outtype: either 'mean', 'unbiased_mean', or 'full'. Mean gives the mean over
#    all ranks for each test label. Unbiased mean accounts for inspection bias (where promiscuous
#    ligands are over-represented in the mean statistic) by first taking the mean rank for EACH 
#    ligand, and then taking mean over all these. 'Full' just returns the ranks of all ligands. 
    """
    if isinstance(test, sparse.csr_matrix):
        test = test.toarray()
    if isinstance(train, sparse.csr_matrix):
        train = train.toarray()
    if isinstance(predictions, sparse.csr_matrix):
        predictions = predictions.toarray()
        
    #This will mask all ROWS that contain no test ligands. No point ranking
    #a row if you're aren't going to evaluate the ranks!
    #(and it works on sparse or np.array)
    row_mask = np.array(test.sum(axis=1)>0).reshape(-1,)
    test_masked = test[row_mask]
    get_ranks = test_masked.astype(bool) #this will select using boolean all test ranks.

    ####Double argsort approach (not used anymore):
    ##order from highest to lowest:
    #order = (-prediction_matrix).argsort(axis=1)
    ##get ranks of each ligand. 
    #ranks = order.argsort(axis=1)

    #This step masks the known positives from the training set,
    #so we are not penalising a highly ranked unknown if it
    #is only behind other true positives. This has a pretty substantial
    #effect since the algo's are really good at ranking known positives highly. 
    predictions = np.ma.masked_array(predictions[row_mask], mask=train[row_mask].astype(bool))    
    #rankdata approach, which correctly handles ties and also thankgod can take masked arrays:
    prediction_ranks = rankdata(-predictions, axis=1)
    
    #all ranks:
    all_test_ranks = prediction_ranks[get_ranks]
    return all_test_ranks

    ##Old stuff:
    #if outtype=='unbiased_mean':
    #    #only calculate mean-rank for ligands having a label (otherwise tonnes of '0' ranks):
    #    m = np.sum(test,axis=1).astype(bool)
    #    #first calculate mean per ligand. Then take mean of all those (avoids inspection bias)
    #    return np.mean(np.mean(np.ma.masked_array(ranks[m], mask=~test[m]), axis=1).data)


def load_subset(subset=False, numchoose=50):
    """
    Loads the 243-target subset of data for HPO. 
    Also removes ligands with only 1 label in the dataset -
    these aren't useful since test/train splits require each 
    instance has at least one label in the training set and
    at least one label in the test set. 
    """
    np.random.seed(100)

    interaction_matrix = sparse.load_npz('../data/interaction_matrix_pchembl.npz')
    interaction_matrix = np.array(interaction_matrix.todense())

    if subset:
        #make smaller to make hyperparameter tuning faster:
        cols= np.random.choice(np.arange(243), numchoose, replace=False)
        interaction_matrix = interaction_matrix[:,cols]
    
    mask = np.sum(interaction_matrix, axis=1)
    mask = mask>1

    interaction_matrix = interaction_matrix[mask] #remove instances with only 1 label
    #possible that some labels don't have ANY instances now... removing columns
    #with no labels:
    interaction_matrix = interaction_matrix[:,np.sum(interaction_matrix, axis=0)>0]
    
    return interaction_matrix
        

def load_time_split(year=2010, return_fingerprints=False):
    """
    Get the interaction matrix of the 243-target subset. Then
    split into two matrices, train and test, based on year of 
    the interactions. 
    :param year: split point. All interactions from this year 
    or afterwards become test interactions. 
    """
    
    interaction_matrix = sparse.load_npz('../data/interaction_matrix_pchembl.npz')
    interaction_dates = sparse.load_npz('../data/interaction_dates_pchembl.npz')

    #turn interaction dates into a masker
    dates_mask = (interaction_dates.data<=year).astype(int)

    #make copies that will become train / test matrices
    train = copy.copy(interaction_matrix)
    test = copy.copy(interaction_matrix)

    #remove entries occuring from `year` and later from train matrix
    train.data = train.data * dates_mask
    #remove all training data from the test matrix. 
    test.data = test.data - train.data

    #remove any rows from the train matrix that have zero interactions.
    #this is the case any time a new ligand is discovered in the cutoff-year or after. 
    #we can't use link prediction on new ligands! It's a cold start problem. 
    #so we remove all these ligands from the present analysis. 
    row_mask = np.array((train.sum(axis=1)!=0)).reshape(1,-1)[0] #there must be a cleaner way to do that.
    train = train[row_mask] 
    test = test[row_mask]
    if return_fingerprints:
        fps = sparse.load_npz('../time_split/morgan.npz')
        fps = fps[row_mask]
        
    #similarly we must now remove any targets that have no data (or not enough) in the training matrix.
    column_mask = (np.array(train.sum(0))[0] >= 20)
    train = train.T[column_mask].T
    test = test.T[column_mask].T
    
    #remove any entries that are explicit zeros because a 0 should be implied by absence
    train.eliminate_zeros()
    test.eliminate_zeros()    

    if return_fingerprints:
        return train, test, fps
    else:
        return train, test


def makeCorrelations(y_in):
    """
    Calculates pairwise correlations between labels.

    :param y_in: interaction matrix - np array of length n_instances and width n_labels
    with 1's in positions of interactions.
    """
    print('y_in shape is:', y_in.shape)
    assert isinstance(y_in, sparse.csr_matrix)
    tot_instances = np.array(y_in.sum(axis=0))[0]

    #shorten interaction matrix to only ligands with 2 or more labels:
    row_mask = np.array(y_in.sum(axis=1)>=2).reshape(1,-1)[0]
    y_in = y_in[row_mask]

    L = sparse.lil_matrix((y_in.shape[1], y_in.shape[1]))

    for idx in tqdm(range(y_in.shape[0]), smoothing=0.1):
        row = y_in[idx]
        if row.sum()>1:
            for j,k in itertools.permutations(row.nonzero()[1], 2):
                L[j,k] += (1)/(tot_instances[k])             
    return L

def train_label_correlation(y_in, L=None):
    """
    Uses the correlation matrix calculated by makeCorrelations to generate
    a matrix of label probabilities
    :param y_in: interaction matrix (csr_matrix)
    :param L: the output of makeCorrelations. A numLabels x numLabels sparse
    array with a value in each position indicating the percentage correlation
    between two labels. 
    """
    if L==None:
        L1 = 1-makeCorrelations(y_in).toarray() #working with dense array is much easier for this. 
                    #but because it's only numLabels x numLabels it's not that big.
                    #(that is, it is not numLabels x numInstances)
    else:
        L1 = 1-L.toarray()
    y_new = y_in.toarray().astype('float32') #working with a dense array again
                                             #for ease of row-wise, elementwise addition 
    for count, row in tqdm(enumerate(y_in), total=y_in.shape[0], smoothing=0.1):
        posLines = row.nonzero()[1]
        corrs = L1[:,posLines]
        probs = 1-np.prod(corrs, axis=1)
        y_new[count]+=probs #elementwise addition here. 
        y_new[count] = np.clip(y_new[count], 0, 1)
    return sparse.csr_matrix(y_new)

##This is the dense version of makeLabelCorrelationPredictions. 
##Faster but unwi
#def makeProbabilities(y, L1):
#    """
#    Uses the correlation matrix from makeCorrelations to create a prediction matrix
#    """
#    y_new = copy.copy(y)
#    for count, row in enumerate(y):
#        posLines = row.nonzero()[0]
#        corrs = L1[:,posLines]
#        probs = 1-np.prod(corrs, axis=1)
#        y_new[count]+=probs
#
#    return clipY(y_new)

def train_implicit_bpr(params, inp):
    model = implicit.bpr.BayesianPersonalizedRanking(factors=params['factors'],
                                                 learning_rate=params['learning_rate'],
                                                 regularization=params['regularization'],
                                                 iterations=params['iterations'],
                                                 use_gpu=False)
    model.fit(sparse.csr_matrix(inp), show_progress=False)
    return np.dot(model.item_factors, model.user_factors.T)

def train_implicit_als(params, inp):
    model = implicit.als.AlternatingLeastSquares(factors=params['factors'],
                                                 regularization=params['regularization'],
                                                 iterations=params['iterations'],
                                                 num_threads=1,
                                                 use_gpu=False)
    model.fit(sparse.csr_matrix(inp), show_progress=False)
    return np.dot(model.item_factors, model.user_factors.T)

def train_implicit_log(params, inp):
    model = implicit.lmf.LogisticMatrixFactorization(factors=params['factors'],
                                                     learning_rate=params['learning_rate'],
                                                     regularization=params['regularization'],
                                                     iterations=params['iterations'],
                                                     num_threads=1,
                                                     use_gpu=False)
    model.fit(sparse.csr_matrix(inp))
    return np.dot(model.item_factors, model.user_factors.T)

def get_lightfm_indexing(inp):
    """Returns user and item indexes used by lightfm. Input is an interaction matrix. """
    return np.arange(inp.shape[0]), np.arange(inp.shape[1])

def train_lightfm_warp(params, inp):
    cid, tid = get_lightfm_indexing(inp)    
    model = lightfm.LightFM(no_components = params['no_components'],
                           loss='warp',
                           max_sampled=params['max_sampled'],
                           learning_rate=params['learning_rate'])
    model.fit(inp, epochs=params['epochs'])
    #get flattened predictions:
    pred_matrix = model.predict(np.repeat(cid, len(tid)), np.tile(tid, len(cid)))
    return np.reshape(pred_matrix, (len(cid), len(tid))) #unflattened.

def train_lightfm_bpr(params, inp):
    cid, tid = get_lightfm_indexing(inp)    
    model = lightfm.LightFM(no_components = params['no_components'],
                           loss='bpr',
                           max_sampled=params['max_sampled'],
                           learning_rate=params['learning_rate'])
    model.fit(inp, epochs=params['epochs'])
    #get flattened predictions:
    pred_matrix = model.predict(np.repeat(cid, len(tid)), np.tile(tid, len(cid)))
    return np.reshape(pred_matrix, (len(cid), len(tid))) #unflattened.

def train_lightfm_log(params, inp):
    cid, tid = get_lightfm_indexing(inp)    
    model = lightfm.LightFM(no_components = params['no_components'],
                           loss='logistic',
                           max_sampled=params['max_sampled'],
                           learning_rate=params['learning_rate'])
    model.fit(inp, epochs=params['epochs'])
    #get flattened predictions:
    pred_matrix = model.predict(np.repeat(cid, len(tid)), np.tile(tid, len(cid)))
    return np.reshape(pred_matrix, (len(cid), len(tid))) #unflattened.

def read_params(name):
    parameter_flag=False
    params = dict()
    for line in open('../hyperparameter_optimization/'+name+'.dat', 'r').readlines():
        if 'Result' in line:
            parameter_flag=False
        if parameter_flag:
            words = line.split()
            if len(words[1])<=5:
                params[words[0]]=int(words[1])
            else:
                params[words[0]]=float(words[1])
        if 'Best parameters' in line:
            parameter_flag=True
    return params


