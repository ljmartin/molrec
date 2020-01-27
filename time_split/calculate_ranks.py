from scipy import sparse
import numpy as np
import itertools
import copy

import sys
sys.path.append("..")
import utils

def calc_ranks_given_name(name, algo, train, test):
    params = utils.read_params(name)
    preds = algo(params, train)
    for _ in range(7):
        preds += algo(params, train)
    ranks = utils.evaluate_predictions(preds, test, train)
    return ranks


if __name__ == '__main__':
    ##Filenames for the algos to load parameters:
    filenames = ['hpo_implicit_als', 'hpo_implicit_bpr',
             'hpo_lightfm_warp', 'hpo_lightfm_bpr']

    ##Functions to train those algorithms:
    algorithms = [utils.train_implicit_als,
                  utils.train_implicit_bpr,
                  utils.train_lightfm_warp,
                  utils.train_lightfm_bpr]
    
    train, test = utils.load_time_split()

    for name, algo in zip(filenames, algorithms):
        print(name)
        ranks = calc_ranks_given_name(name, algo, train, test)
        np.save(name+'.npy', ranks)

