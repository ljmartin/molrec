import matplotlib.pyplot as plt
import pymc3 as pm
from scipy.stats import gaussian_kde
import numpy as np 

plt.style.use('seaborn-colorblind')

def calc_hpd(ranks, statistic=np.mean):
    with pm.Model() as model:
        #prior on statistic of interest:
        a = pm.Normal('a', mu=statistic(ranks), sigma=10.0)
        #'nuisance' parameter:
        b = pm.HalfNormal('b', sigma=10.0)
        #likelihood:
        if statistic==np.mean:
            y = pm.Normal('y', mu=a, sigma=b, observed=ranks)
        elif statistic==np.median:
            y = pm.Laplace('y', mu=a, b=b,observed=ranks)

        trace = pm.sample(draws=500, tune=500, chains=2, target_accept=0.9)
    return statistic(ranks), pm.stats.hpd(trace['a'])
    
def calc_kde(ranks):
    #kde:
    density = gaussian_kde(ranks)
    xs = np.linspace(0,243,243)
    density.covariance_factor= lambda : 0.25
    density._compute_covariance()
    return density(xs)

def calc_ecdf(ranks):
    ecdf = [(ranks<i).sum()/len(ranks) for i in range(0, 243)]
    return ecdf

if __name__ == '__main__':
    ##Filenames for the algos to load parameters:
    filenames = ['hpo_implicit_als', 'hpo_implicit_bpr',
             'hpo_lightfm_warp', 'hpo_lightfm_bpr']


    ##Plot first figure:
    fig, ax = plt.subplots(2)
    fig.set_figheight(8)
    fig.set_figwidth(8)
    
    for count, name in enumerate(filenames):
        ranks = np.load(name+'.npy')
        mean, mean_hpd = calc_hpd(ranks, np.mean)
        median, median_hpd = calc_hpd(ranks, np.median)
        ax[0].bar(count, mean, label=name)
        ax[0].errorbar(count, mean, 
                       yerr=[[mean-mean_hpd[0]],[mean_hpd[1]-mean]], 
                       color='black', elinewidth=2, capsize=3, markeredgewidth=2)
        ax[1].bar(count, median)
        ax[1].errorbar(count, median, 
                       yerr=[[median-median_hpd[0]],[median_hpd[1]-median]], 
                       color='black', elinewidth=2, capsize=3, markeredgewidth=2)

    ax[0].set_ylabel('Mean rank', fontsize=20)
    ax[0].set_xticks([])
    ax[0].legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, ncol=2)

    ax[1].set_ylabel('Median rank', fontsize=20)
    ax[1].set_xticks([])

    plt.tight_layout()
    fig.savefig('statistics.eps')
    plt.close(fig)

    fig = plt.figure(figsize=(12,6))
    grid = plt.GridSpec(2, 5, wspace=0.1, hspace=0.4)
    ax1 = fig.add_subplot(grid[0, :3])
    ax2 = fig.add_subplot(grid[0, 3:])
    ax3 = fig.add_subplot(grid[1, :3])
    ax4 = fig.add_subplot(grid[1, 3:]);
    ax = [ax1, ax2, ax3, ax4]

    for name in filenames:
        ranks = np.load(name+'.npy')
        kde = calc_kde(ranks)
        ecdf = calc_ecdf(ranks)
        ax1.plot(kde, label=name)
        ax2.plot(kde, label=name)
        ax3.plot(ecdf, label=name)
        ax4.plot(ecdf, label=name)

    ax1.set_xlim(0,243)
    ax1.set_title('Ranks KDE')
    ax1.set_ylabel('Density', fontsize=14)
    ax1.yaxis.grid()
    
    ax2.set_xlim(0,20)
    ax2.set_title('Ranks KDE (top 20)')
    ax2.yaxis.grid()

    ax3.set_xlim(0,243)
    ax3.set_title('Ranks ECDF')
    ax3.set_ylabel('Cumulative density', fontsize=14)
    ax3.yaxis.grid()

    ax4.set_xlim(0,20)
    ax4.set_title('Ranks ECDF (top 20)')
    ax4.yaxis.grid()

    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)
    ax1.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)

    fig.savefig('distributions.eps')
    plt.close(fig)

    
 


