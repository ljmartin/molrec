import tqdm
import numpy as np
from scipy import sparse

import sys
sys.path.append("..")
import utils

from joblib import Parallel, delayed
from scipy.spatial.distance import squareform

import pymc3 as pm

            
if __name__=="__main__":
    #load time split and fingerprints:
    train, test, fps = utils.load_time_split(year=2015, return_fingerprints=True)

    sea = SEA(train, fps)
    sea.fit()
    sea.predict()
    
    ranks = utils.evaluate_predictions(sea.y_new, test, train)

    np.save('./processed_data/2015_sea.npy', ranks)
