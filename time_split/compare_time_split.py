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
    f = open('../hyperparmeter_optimization/'+filename, 'r')
    record=False
    for line in f:
        if 'Paramaters:' in line:
            return [float(i) for i in line.strip('\n').replace('[', '').replace(']', '').replace(',', '').split()[1:]]


##Implicit:
#bpr:
pars = find_opt_pars(filenames[1])
print(pars)
model = implicit.bpr.BayesianPersonalizedRanking(factors=int(pars[0]), 
                                                 learning_rate=pars[1],
                                                 regularization=pars[2],
                                                 iterations=int(pars[3]),
                                                 use_gpu=False)

model.fit(sparse.csr_matrix(train))
prediction_matrix=np.dot(model.item_factors, model.user_factors.T)
test_ranks = utils.evaluate_predictions(prediction_matrix, sparse.csr_matrix(test),avg=False)
print(np.mean(test_ranks), np.median(test_ranks))
