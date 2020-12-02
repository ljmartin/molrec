from scipy import sparse
import numpy as np
import itertools
import copy

import sys
sys.path.append("..")
import utils

def calc_ranks_given_name(name, algo, train, test, fps=None):
    if name == 'label_correlation':
        preds = algo(train)
        ranks = utils.evaluate_predictions(preds, test, train)
        return ranks
    params = utils.read_params(name)

    if name =='sea':
        preds = algo(params, train, fps, njobs=8, fraction=2)
    elif name =='rfc':
        preds = algo(params, train, fps, njobs=8)
    else: #these are the recommender algorithms
        preds = algo(params, train)
        
    ranks = utils.evaluate_predictions(preds, test, train)
    return ranks


if __name__ == '__main__':
    ##Filenames for the algos to load parameters:
    filenames = ['label_correlation', 'hpo_implicit_bpr', 'hpo_lightfm_warp',
                 'sea', 'rfc']

    ##Functions to train those algorithms:
    algorithms = [utils.train_label_correlation,
                  utils.train_implicit_bpr,
                  utils.train_lightfm_warp,
                  utils.train_sea,
                  utils.train_rfc]



    yrs = [2010, 2011, 2012, 2013, 2014, 2015, 2016]

    for year in yrs:
        train, test, fps = utils.load_time_split(year=year, return_fingerprints=True)

        for name, algo in zip(filenames, algorithms):
            if name =='rfc':
                pass
            else:
                continue
            print(name)                
            ranks = calc_ranks_given_name(name, algo, train, test, fps)
            np.save('./processed_data/'+str(year)+'_'+name+'.npy', ranks)

