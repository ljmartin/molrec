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
    ax.text(0, 1.1, lab, transform=ax.transAxes,
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

def plot(nn=False):
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
    
        #analyse
        bstrap = simple_bootstrap(ranks, take=len(ranks))
        ci = simple_ci(bstrap)
        x,y = simple_ecdf(ranks, 243)

        print(name, np.mean(bstrap), ci)
    
        #plot:
        #A
        ax[0].plot(x,y,'-o', mfc='white', mew=1.5, linewidth=1.5,label=name, c='C'+str(count))
    
        #B
        sjitter = np.abs(np.random.randn(len(bstrap))) / 10
        ljitter = 0#np.random.randn(len(bstrap))/200
        ax[1].scatter(count+sjitter+0.15, bstrap+ljitter, alpha=0.05)
        ax[1].plot([count,count], [ci[0], ci[1]], lw=5.5, c='C'+str(count),zorder=10)
        ax[1].scatter([count], [np.mean(ci)],
                       facecolor='white',
                       edgecolor='C'+str(count),
                       lw=3.5,
                       zorder=20)

    fsize = 14
    ax[1].set_ylim(-0.05,0.6)
    ax[1].set_xticks([0,1,2,3,4,5])
    ax[1].set_xticklabels([i.replace("hpo_", '') for i in filenames], rotation=35, ha='center',fontsize=fsize)
    ax[1].set_ylabel('p@3', fontsize=fsize)
    yt = np.arange(-5,25, 5)
    #ax[1].set_yticks(yt)
    #ax[1].set_yticklabels(['' for i in yt])
    
    plot_fig_label(ax[0], 'A.')
    plot_fig_label(ax[1], 'B.')


    ax[0].set_xlim(0,20)
    ax[0].set_ylim(-0.05, 1)
    ax[0].set_xlabel('Rank', fontsize=fsize)
    ax[0].set_ylabel('ECDF', fontsize=fsize)
    xt = np.arange(1,20,2)
    ax[0].set_xticks(xt)
    ax[0].axvline(3, c='k', linestyle='--', zorder=1)
    ax[1].axhline(0, c='k', linestyle='--', alpha=0.2)


    ax[1].plot([0,0], [1e6, 1e6+1], lw=5.5, c='k',zorder=10, label='95% CI')
    ax[1].scatter([0], [1e6], c='k', alpha=0.6, label='Bootstrap estimates')
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()

    return fig, ax

fig, ax = plot()
fig.savefig('./figures/timesplit2.png')
fig.savefig('./figures/timesplit2.tif')
fig.savefig('./figures/timesplit2.svg')
fig, ax = plot(nn=True)
fig.savefig('./figures/timesplit_minusNN2.png')
fig.savefig('./figures/timesplit_minusNN2.tif')
fig.savefig('./figures/timesplit_minusNN2.svg')
