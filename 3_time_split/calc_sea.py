import tqdm
import numpy as np
from scipy import sparse

import sys
sys.path.append("..")
import utils

from joblib import Parallel, delayed
from scipy.spatial.distance import squareform

import pymc3 as pm

class SEA(object):
    def __init__(self, interaction_matrix, fps):
        self.interaction_matrix = interaction_matrix
        self.fps = fps
        self.fp_dict = {}
        for i in range(self.interaction_matrix.shape[1]):
            mask = (self.interaction_matrix[:,i]==1).toarray().flatten()
            self.fp_dict[i] = self.fps[mask][np.random.choice(mask.sum(), mask.sum()//5, replace=False)]
        
    def fast_jaccard(self,X, Y=None):
        """credit: https://stackoverflow.com/questions/32805916/compute-jaccard-distances-on-sparse-matrix"""
        if isinstance(X, np.ndarray):
            X = sparse.csr_matrix(X)
        if Y is None:
            Y = X
        else:
            if isinstance(Y, np.ndarray):
                Y = sparse.csr_matrix(Y)
        assert X.shape[1] == Y.shape[1]

        X = X.astype(bool).astype(int)
        Y = Y.astype(bool).astype(int)
        intersect = X.dot(Y.T)
        x_sum = X.sum(axis=1).A1
        y_sum = Y.sum(axis=1).A1
        xx, yy = np.meshgrid(x_sum, y_sum)
        union = ((xx + yy).T - intersect)
        return (1 - intersect / union).A
        
    def generate_random_pairwise_comparison(self):
        idx_1 = np.random.choice(self.fps.shape[0], np.random.choice(np.arange(20,1000)))
        idx_2 = np.random.choice(self.fps.shape[0], np.random.choice(np.arange(20,1000)))
        rss = self.fast_jaccard(self.fps[idx_1], self.fps[idx_2])
        num_comparisons = rss.shape[0]*rss.shape[1]
        return num_comparisons, rss[rss>0.3].sum()
    
    def generate_parameters(self, n=200):
        ns = list()
        rs = list()
        for i in range(n):
            n, r = self.generate_random_pairwise_comparison()
            ns.append(n)
            rs.append(r)

        with pm.Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
            # Define priors
            sigma = pm.HalfNormal('sigma', sigma=1000)
            error_scaler = pm.HalfNormal('sigma_scaler',sigma=1000)
            x_coeff = pm.Normal('x', 0, sigma=200)
            intercept = pm.Normal('Intercept', 0, sigma=200)
            #  Define likelihood
            likelihood = pm.Normal('y', mu=intercept + x_coeff * np.array(ns),
                        sigma=sigma+error_scaler*np.array(ns), observed=np.array(rs))
            # Inference!
            trace = pm.sample(2000, tune=2000, target_accept =0.9, cores=2) 
        self.gradient = trace['x'].mean()
        self.sigma = trace['sigma'].mean()
        self.sigma_scaler = trace['sigma_scaler'].mean()
        
    def _rss(self, targs):
        targ1 = targs[0]
        targ2 = targs[1]
        distances = self.fast_jaccard(self.fp_dict[targ1], self.fp_dict[targ2])
        num_comparisons = distances.shape[0]*distances.shape[1]
        
        return num_comparisons, distances[distances>0.3].sum()
    
    def _wrap_rss(self, idx):
        return [self._rss(i) for i in idx]
    
    def calc_rss_matrix(self, njobs=8):
        
        num_cols = self.interaction_matrix.shape[1]
        j_,k_= np.triu_indices(num_cols,k=1)
        
        idx = np.array([(j_[i], k_[i]) for i in range(len(j_))])
        parallel_jobs = Parallel(n_jobs=njobs)(delayed(self._wrap_rss)(i) for i in np.array_split(idx, njobs))
        
        raw_scores = list()
        num_comparisons = list()
        
        for job in parallel_jobs:
            for item in job:
                num_comparisons.append(item[0])
                raw_scores.append(item[1])
                
        self.raw_scores = squareform(raw_scores)
        self.num_comparisons = squareform(num_comparisons)
            
    def calc_pP(self):
        e_m = 0.577215665 #euler mascheroni
        
        self.Z_scores = (self.raw_scores - (self.num_comparisons * self.gradient)) \
            / (self.sigma + self.sigma_scaler*self.num_comparisons)

        def x(z): # taylor expansion of p value calculation
            return -np.exp( (-z*np.pi) / (np.sqrt(6) - e_m))
        
        pz = -(x(self.Z_scores) + x(self.Z_scores)**2 / 2 + x(self.Z_scores)**3 / 3)
        
        self.pz = -np.log(pz)
        
        
    def fit(self):
        self.calc_rss_matrix()
        self.generate_parameters()
        self.calc_pP()
        
    def predict(self):
        self.y_new = self.interaction_matrix.copy().toarray()
        for count, row in tqdm.tqdm(enumerate(self.interaction_matrix), 
                                               total=self.interaction_matrix.shape[0], 
                                               smoothing=0.1):
            posLines = row.nonzero()[1]
            corrs = self.pz[:,posLines]
            probs = 1-np.sum(corrs, axis=1)
            self.y_new[count]=probs
        



            
if __name__=="__main__":
    #load time split and fingerprints:
    train, test, fps = utils.load_time_split(year=2015, return_fingerprints=True)

    sea = SEA(train, fps)
    sea.fit()
    sea.predict()
    
    ranks = utils.evaluate_predictions(sea.y_new, test, train)

    np.save('./processed_data/2015_sea.npy', ranks)
