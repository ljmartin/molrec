#normal tools:
from scipy import sparse
import numpy as np
import copy
import sys
sys.path.append("..")
import utils

#learning libraries:
import implicit
import lightfm



##load data:
train, test = utils.load_time_split()

filenames = ['hpo_implicit_als.dat', 'hpo_implicit_bpr.dat',
             'hpo_lightfm_warp.dat', 'hpo_lightfm_bpr.dat']
def find_opt_pars(filename):
    f = open('../hyperparameter_optimization/'+filename, 'r')
    record=False
    for line in f:
        if 'Paramaters:' in line:
            return [float(i) for i in line.strip('\n').replace('[', '').replace(']', '').replace(',', '').split()[1:]]

        


def train_implicit_als(pars):
    model = implicit.als.AlternatingLeastSquares(factors=int(pars[0]),
                                                     regularization=pars[1],
                                                     iterations=int(pars[2]),
                                                     num_threads=1,
                                                     use_gpu=False)
    model.fit(sparse.csr_matrix(train))
    prediction_matrix=np.dot(model.item_factors, model.user_factors.T)
    test_ranks = utils.evaluate_predictions(prediction_matrix, sparse.csr_matrix(test),avg=False)
    return np.mean(test_ranks), np.median(test_ranks)

def train_implicit_bpr(pars):
    model = implicit.bpr.BayesianPersonalizedRanking(factors=int(pars[0]), 
                                                 learning_rate=pars[1],
                                                 regularization=pars[2],
                                                 iterations=int(pars[3]),
                                                 use_gpu=False)

    model.fit(sparse.csr_matrix(train))
    prediction_matrix=np.dot(model.item_factors, model.user_factors.T)
    test_ranks = utils.evaluate_predictions(prediction_matrix, sparse.csr_matrix(test),avg=False)
    return np.mean(test_ranks), np.median(test_ranks)



##LightFM:
#lightfm 'user id' (chemical id)                                                                                       
cid = np.arange(train.shape[0])
#lightfm 'item id' (target id)                                                                                         
tid = np.arange(train.shape[1])



def train_lightfm_warp(pars):
    model = lightfm.LightFM(no_components = int(pars[0]),
                           loss='warp', 
                           max_sampled=int(pars[1]),
                           learning_rate=pars[2])
    model.fit(sparse.csr_matrix(train), epochs=int(pars[3]))
    prediction_matrix = model.predict(np.repeat(cid, len(tid)), np.tile(tid, len(cid)))
    prediction_matrix = np.reshape(prediction_matrix, (len(cid), len(tid)))
    test_ranks = utils.evaluate_predictions(prediction_matrix, sparse.csr_matrix(test),avg=False)
    return np.mean(test_ranks), np.median(test_ranks)

def train_lightfm_bpr(pars):
    model = lightfm.LightFM(no_components = int(pars[0]),
                           loss='bpr', 
                           max_sampled=int(pars[1]),
                           learning_rate=pars[2])
    model.fit(sparse.csr_matrix(train), epochs=int(pars[3]))
    prediction_matrix = model.predict(np.repeat(cid, len(tid)), np.tile(tid, len(cid)))
    prediction_matrix = np.reshape(prediction_matrix, (len(cid), len(tid)))
    test_ranks = utils.evaluate_predictions(prediction_matrix, sparse.csr_matrix(test),avg=False)
    return np.mean(test_ranks), np.median(test_ranks)
    

outfile = open('results.dat', 'w')
outfile.write('algorithm, mean, median\n')

##implicit:
#als:
pars = find_opt_pars(filenames[0])
for _ in range(3):
    mean, median = train_implicit_als(pars)
    outfile.write(filenames[0]+': '+str(mean)+' '+str(median)+'\n')

#bpr:
pars = find_opt_pars(filenames[1])
for _ in range(3):
    mean, median = train_implicit_bpr(pars)
    outfile.write(filenames[1]+': '+str(mean)+' '+str(median)+'\n')

##lightfm
#warp
pars = find_opt_pars(filenames[2])
for _ in range(3):
    mean, median = train_lightfm_warp(pars)
    outfile.write(filenames[2]+': '+str(mean)+' '+str(median)+'\n')

#bpr
pars = find_opt_pars(filenames[3])
for _ in range(3):
    mean, median = train_lightfm_bpr(pars)
    outfile.write(filenames[3]+': '+str(mean)+' '+str(median)+'\n')
    
#label correlation:
L1 = 1- utils.makeCorrelations(train)
prediction_matrix = utils.makeProbabilities(train, L1)
test_ranks = utils.evaluate_predictions(prediction_matrix, sparse.csr_matrix(test), avg=False)
outfile.write('label correl: '+str(np.mean(test_ranks))+' '+str(np.median(test_ranks))+'\n')


outfile.close()
