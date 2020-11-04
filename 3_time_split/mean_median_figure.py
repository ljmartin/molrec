import matplotlib.pyplot as plt
import pymc3 as pm
from scipy.special import logit, expit
from scipy.stats import gaussian_kde, laplace, norm
import numpy as np 
plt.style.use('seaborn')


filenames = ['label_correlation', 'hpo_implicit_als', 'hpo_implicit_bpr',
             'hpo_lightfm_warp', 'hpo_lightfm_bpr', 'nearest_neighbor']

yrs = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
num_targets = [225, 228, 231, 234, 237, 240, 242, 243, 243, 243]
year = 2015

def plot_fig_label(ax, lab):
    ax.text(0, 1.15, lab, transform=ax.transAxes,
        fontsize=24, va='top', ha='left')

def simple_ci(data):
    d_sorted = np.sort(data)
    low = int(d_sorted.shape[0] // (1/0.025))
    high = int(d_sorted.shape[0] // (1/0.975))
    return (d_sorted[low], d_sorted[high])

def plot_meanmed(nn=False):
    def simple_bootstrap(data, dist=norm, n=1000, take = 350):
        samples = np.random.choice(data, size=(n,take))
        estimates = [dist.fit(i)[0] for i in samples]
        return np.array(estimates)
    
    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.set_figheight(10)
    fig.set_figwidth(7)
    if nn:
        nnranks = np.load('./processed_data/2015_nearest_neighbor.npy')
        mask = nnranks>3
    
    for count, name in enumerate(filenames):
        #load
        ranks = np.load('./processed_data/'+str(year)+'_'+name+'.npy')
        if nn:
            ranks = ranks[mask]
    

        for a, fun in zip([0,1], [norm, laplace]):
            #analyse
            #logit transform ranks:
            logit_ranks = logit(ranks / 243)
            
            bstrap = expit(simple_bootstrap(logit_ranks, fun, take=len(ranks)))*243
            ci = simple_ci(bstrap)

            sjitter = np.abs(np.random.randn(len(bstrap))) / 10
            ljitter = np.random.randn(len(bstrap))/20
            ax[a].scatter(count+sjitter+0.15, bstrap+ljitter, alpha=0.05)
            
            
            ax[a].plot([count,count], [ci[0], ci[1]], lw=5.5, c='C'+str(count),zorder=10)
            ax[a].scatter([count], [np.mean(ci)], 
                       facecolor='white', 
                       edgecolor='C'+str(count),
                       lw=3.5,
                       zorder=20)
    
    fsize = 14
    
    
    for a in ax:
        a.plot([0,0], [1e6, 1e6+1], lw=5.5, c='k',zorder=10, label='95% CI')
        a.scatter([0], [1e6], c='k', alpha=0.6, label='Bootstrap estimates')
    
    ax[1].set_ylim(0,26)
    ax[1].legend()
    ax[1].set_ylabel('Median rank', fontsize=fsize)
    
    ax[1].set_xticks([0,1,2,3,4,5])
    ax[1].set_xticklabels([i.replace("hpo_", '') for i in filenames], rotation=35, ha='center',fontsize=fsize)
    
    ax[0].set_xticks([0,1,2,3,4,5])
    ax[0].set_xticklabels(['' for _ in filenames], rotation=65, ha='center', fontsize=fsize)


    plot_fig_label(ax[0], 'A.')
    plot_fig_label(ax[1], 'B.')

    ax[0].legend()
    ax[0].set_ylim(0,25)
    ax[0].set_xlabel('', fontsize=fsize)
    ax[0].set_ylabel('Mean rank', fontsize=fsize)
    
    
    plt.tight_layout()

    return fig, ax



fig, ax = plot_meanmed()
fig.savefig('./figures/mean_median.png')
fig.savefig('./figures/mean_median.tif')
fig.savefig('./figures/mean_median.svg')
