import matplotlib.pyplot as plt
import pymc3 as pm
from scipy.stats import gaussian_kde
import numpy as np 
plt.style.use('seaborn')
from statsmodels.distributions import ECDF

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

    return trace
    
def calc_kde(ranks, xs=np.linspace(0,243,243)):
    #kde:
    density = gaussian_kde(ranks)
    density.covariance_factor= lambda : 0.25
    density._compute_covariance()
    return density(xs)

#def calc_ecdf(ranks):
#    ecdf = [(ranks<i).sum()/len(ranks) for i in range(0, 243)]
#    return ecdf

def plot_fig_label(ax, lab):
    ax.text(0, 1.15, lab, transform=ax.transAxes,
        fontsize=24, va='top', ha='left')

if __name__ == '__main__':
    ##First we will take the three best performing algorithms and 
    ##take the geometric average of their rankings:
    #ranks = [np.load(name+'.npy') for name in ['label_correlation', 'hpo_implicit_bpr', 'hpo_lightfm_warp']]
    #geo_avg = np.power(ranks[0]*ranks[1]*ranks[2], 1/3)
    #np.save('geometric_avg', geo_avg)
    
    ##Now we can proceed to graph all the rankings:
    ##Filenames for the algos to load parameters:
    #filenames = ['geometric_avg', 'label_correlation', 'hpo_implicit_als', 'hpo_implicit_bpr',
    #         'hpo_lightfm_warp', 'hpo_lightfm_bpr']

    filenames = ['label_correlation', 'hpo_implicit_als', 'hpo_implicit_bpr',
             'hpo_lightfm_warp', 'hpo_lightfm_bpr']



    ##Plot first figure:
    fig, ax = plt.subplots(2)
    fig.set_figheight(8)
    fig.set_figwidth(8)
    
    for count, name in enumerate(filenames):
        ranks = np.load(name+'.npy')
        mean_trace = calc_hpd(ranks, np.mean)
        median_trace = calc_hpd(ranks, np.median)
        print(name)
        for j,trace in zip([0,1], [mean_trace, median_trace]):
            m = np.mean(trace['a'])
            hpd = pm.hpd(trace['a'])
            print(m, hpd)
            xs = np.linspace(m-3,m+3,100)
            density = calc_kde(trace['a'], xs=xs)
        
            ax[j].errorbar(count, m, yerr = np.array([m-hpd[0], hpd[1]-m])[:,None],
                           fmt='o', linewidth=4, markersize=15, capsize=3)
            ax[j].fill_betweenx(xs,density+count,count, alpha=0.4,label=name.strip('hpo_'))

    ax[0].set_ylabel('Mean rank', fontsize=20)
    ax[0].set_xticks([])
    ax[0].legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, ncol=2)

    ax[1].set_ylabel('Median rank', fontsize=20)
    ax[1].set_xticks([])

    plot_fig_label(ax[0], 'A')
    plot_fig_label(ax[1], 'B')

    plt.tight_layout()
    fig.savefig('statistics.pdf')
#    fig.savefig('statistics.svg')
    plt.close(fig)


    ##Plot second figure:
    fig = plt.figure(figsize=(12,6))
    grid = plt.GridSpec(2, 5, wspace=0.25, hspace=0.4)
    ax1 = fig.add_subplot(grid[0, :3])
    ax2 = fig.add_subplot(grid[0, 3:])
    ax3 = fig.add_subplot(grid[1, :3])
    ax4 = fig.add_subplot(grid[1, 3:]);
    ax = [ax1, ax2, ax3, ax4]


    for name in filenames:
        ranks = np.load(name+'.npy')

        ##Plot histogram:
        n, x = np.histogram(ranks, bins = np.linspace(0,243,243))
        bin_centers = 0.5*(x[1:]+x[:-1])
        ax1.plot(bin_centers,n, label=name)
        ax2.plot(bin_centers,n, label=name, linewidth=0.5)
 
        ##Plot empirical cumulative distribution function
        ecdf = ECDF(ranks)
        ax3.plot(ecdf.x-0.5,ecdf.y)
        ax4.plot(ecdf.x-0.5,ecdf.y, linewidth=0.5)
        if name == 'label_correlation':
            ax4.plot([5,5],[0,ecdf.y[ecdf.x==5][-1]], c='k', linestyle='--')
            ax4.plot([0,5],[ecdf.y[ecdf.x==5][-1],ecdf.y[ecdf.x==5][-1]],c='k', linestyle='--', label='Label correlation\nECDF at rank 5')

    ax1.set_xlim(0,243)
    ax1.set_title('Histogram of predicted ranks')
    ax1.set_ylabel('Count density', fontsize=14)
    ax1.yaxis.grid()
    ax1.axvline(20, linestyle='--', c='k', label='Rank 20')
    plot_fig_label(ax1, 'A')
    
    ax2.set_xlim(1,20)
    ax2.set_title('Ranks histogram (top 20)')
    #ax2.yaxis.grid()
    ax2.set_xticks(np.arange(0,21,2))
    plot_fig_label(ax2, 'B')

    ax3.set_xlim(0,243)
    ax3.set_title('Empirical CDF of predicted ranks')
    ax3.set_ylabel('Cumulative\nnormalized density', fontsize=14)
    ax3.yaxis.grid()
    ax3.axvline(20, linestyle='--', c='k')
    plot_fig_label(ax3, 'C')

    ax4.set_xlim(1,20)
    ax4.set_title('Ranks empirical CDF (top 20)')
    #ax4.yaxis.grid()
    ax4.legend()
    ax4.set_xticks(np.arange(0,21,2))
    plot_fig_label(ax4, 'D')

    #plt.setp(ax2.get_yticklabels(), visible=False)
    #plt.setp(ax4.get_yticklabels(), visible=False)
    ax1.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)

    fig.savefig('distributions.pdf')
#    fig.savefig('distributions.svg')
    plt.close(fig)

    
 


