from scipy import sparse
import numpy as np
import itertools
import copy

import sys
sys.path.append("..")
import utils

def calc_ranks_given_name(name, algo, train, test):
    if name == 'label_correlation':
        preds = algo(train)
    else:
        params = utils.read_params(name)
        preds = algo(params, train)
        for _ in range(6):
            preds += algo(params, train)
    ranks = utils.evaluate_predictions(preds, test, train)
    return ranks


if __name__ == '__main__':
    ##Filenames for the algos to load parameters:
    filenames = ['label_correlation', 'hpo_implicit_als', 'hpo_implicit_bpr',
             'hpo_lightfm_warp', 'hpo_lightfm_bpr']

    ##Functions to train those algorithms:
    algorithms = [utils.train_label_correlation,
                  utils.train_implicit_als,
                  utils.train_implicit_bpr,
                  utils.train_lightfm_warp,
                  utils.train_lightfm_bpr]

    yrs = [2010, 2011, 2012, 2013, 2014, 2015, 2016]

    for year in yrs:
        train, test = utils.load_time_split(year=year)

        for name, algo in zip(filenames, algorithms):
            print(name)
            ranks = calc_ranks_given_name(name, algo, train, test)
            np.save('./processed_data/'+str(year)+'_'+name+'.npy', ranks)

