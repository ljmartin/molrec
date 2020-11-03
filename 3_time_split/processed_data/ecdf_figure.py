import matplotlib.pyplot as plt
from scipy.special import logit, expit
from scipy.stats import gaussian_kde
import numpy as np 
plt.style.use('seaborn')
from statsmodels.distributions import ECDF

from seaborn import kdeplot



filenames = ['label_correlation', 'hpo_implicit_als', 'hpo_implicit_bpr',
             'hpo_lightfm_warp', 'hpo_lightfm_bpr', 'nearest_neighbor']

yrs = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
num_targets = [225, 228, 231, 234, 237, 240, 242, 243, 243, 243]
year = 2015

def plot_fig_label(ax, lab):
    ax.text(0, 1.15, lab, transform=ax.transAxes,
        fontsize=24, va='top', ha='left')

def simple_bootstrap(data, n=1000, take = 350):
    return (np.random.choice(data, size=(n,take))<=3).sum(1) /take

def simple_ci(data):
    d_sorted = np.sort(data)
    low = int(d_sorted.shape[0] // (1/0.025))
    high = int(d_sorted.shape[0] // (1/0.975))
    return (d_sorted[low], d_sorted[high])

def simple_ecdf(ranks, maxrank):
    x = np.arange(1, maxrank)
    ecdf = [(ranks<=i).sum()/len(ranks) for i in x]
    return x, ecdf

