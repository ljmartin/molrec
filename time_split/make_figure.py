import matplotlib.pyplot as plt
import pymc3 as pm
from scipy.special import logit, expit
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
    filenames = ['label_correlation', 'hpo_implicit_als', 'hpo_implicit_bpr',
             'hpo_lightfm_warp', 'hpo_lightfm_bpr']

    yrs = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    num_targets = [225, 228, 231, 234, 237, 240, 242, 243, 243, 243]


    #plot supplementary figure describing which year to use for time split
    fig = plt.figure()
    ax1 = plt.subplot(231)
    ax2 = plt.subplot(232)
    ax3 = plt.subplot(233)
    ax4 = plt.subplot(234)
    ax5 = plt.subplot(235)
    from scipy.stats import sem 
    fig.set_figheight(15)
    fig.set_figwidth(15)
    z = 1.959964
    for name, a in zip(filenames, [ax1, ax2, ax3, ax4, ax5]):
        lows = list()
        highs = list()
        middles = list()
        for count, year, num in zip(range(len(yrs)), yrs, num_targets):
            ranks = np.load('./processed_data/'+str(year)+'_'+name+'.npy')/num
            log_ranks = np.log10(ranks)
            s = sem(log_ranks)
            m = log_ranks.mean()
            low = 10**(m-s*z)*num
            high = 10**(m+s*z)*num
            highs.append(high)
            lows.append(low)
            middles.append(10**m*num)
        a.fill_between(yrs, y1=lows, y2=highs, label=name)
        a.plot(yrs, middles,  '-o', c='white',)
        a.set_ylim(1, 30)
        a.set_title(name)
    fig.savefig('supp.pdf')
    fig.savefig('supp.tif')

    
    ##Plot first figure:
    fig, ax = plt.subplots(2)
    fig.set_figheight(8)
    fig.set_figwidth(8)

    year = 2015
    for count, name in enumerate(filenames):
        ranks = np.load('./processed_data/'+str(year)+'_'+name+'.npy')
        
        logit_transformed_ranks = logit(ranks/243)

        mean_trace = calc_hpd(logit_transformed_ranks, np.mean)
        median_trace = calc_hpd(logit_transformed_ranks, np.median)
        print(name)
        for j,trace in zip([0,1], [mean_trace, median_trace]):
            untransformed_samples = expit(trace['a'])*244
            m = np.mean(untransformed_samples)
            hpd = pm.hpd(untransformed_samples)
            print(m, hpd)
            xs = np.linspace(m-3,m+3,100)
            density = calc_kde(untransformed_samples, xs=xs)/2
        
            ax[j].errorbar(count, m, yerr = np.array([m-hpd[0], hpd[1]-m])[:,None],
                           fmt='o', mfc='white', mew=2, linewidth=4, markersize=7.5, capsize=3)
            ax[j].fill_betweenx(xs,density/2+count,count, alpha=0.4,label=name.strip('hpo_'))

    ax[0].set_ylabel('Mean rank', fontsize=20)
    ax[0].set_xticks([])
    ax[0].legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, ncol=2)

    ax[1].set_ylabel('Median rank', fontsize=20)
    ax[1].set_xticks([])

    plot_fig_label(ax[0], 'A')
    plot_fig_label(ax[1], 'B')

    plt.tight_layout()
    fig.savefig('statistics.pdf')
    fig.savefig('statistics.tif')
    plt.close(fig)


    ##Plot second figure:
    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.set_figheight(6)
    fig.set_figwidth(12)
    ax1 = ax[0,0]
    ax2 = ax[0,1]
    ax3 = ax[1,0]
    ax4 = ax[1,1]


    for name in filenames:
        ranks = np.load('./processed_data/'+str(year)+'_'+name+'.npy')

        ##Plot histogram:
        n, x = np.histogram(ranks, bins = np.linspace(1,244,244))
        ax1.plot(x[:-1]+np.random.uniform(-0.15,0.15,len(n)),n, label=name)
        ax2.plot(x[:-1]+np.random.uniform(-0.15,0.15,len(n)),n,'-o', mfc='white', mew=1.5, label=name, linewidth=0.5)

        ##Plot empirical cumulative distribution function
        ecdf = np.cumsum(n)/n.sum()
        ax3.plot(x[:-1]+np.random.uniform(-0.1,0.1,len(n)),ecdf)
        ax4.plot(x[:-1]+np.random.uniform(-0.1,0.1,len(n)),ecdf, '-o', mfc='white', mew=1.5, linewidth=0.5)
        if name == 'label_correlation':
            ax4.plot([0,3],[ecdf[2],ecdf[2]],c='C0', linestyle=':',label='Label correlation\nECDF at rank 3')


    ax1.set_title('Histogram of predicted ranks')
    ax1.set_ylabel('Count density')
    ax1.yaxis.grid()
    ax1.axvline(20, linestyle='--', c='k', label='Rank 20')
    ax1.set_xlim(0,244)
    plot_fig_label(ax1, 'A')
    
    ax2.set_xlim(0,21)
    ax2.set_title('Histogram, top 20')
    ax2.set_xticks(np.arange(1,21,1))
    plot_fig_label(ax2, 'B')
    
    ax3.set_xlim(0,244)
    ax3.set_title('Empirical CDF (ECDF) of predicted ranks')
    ax3.set_ylabel('Cumulative\nnormalized density')
    ax3.yaxis.grid()
    ax3.axvline(20, linestyle='--', c='k')
    ax3.set_xlabel('Ranks')
    plot_fig_label(ax3, 'C')

    ax4.set_xlim(0,21)
    ax4.set_ylim(0.1, 0.7)
    ax4.set_title('ECDF, top 20')
    ax4.legend()
    ax4.set_xticks(np.arange(1,21,1))
    ax4.set_xlabel('Ranks')
    plot_fig_label(ax4, 'D')

    ax1.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.tight_layout()

    fig.savefig('distributions.pdf')
    fig.savefig('distributions.tif')
    plt.close(fig)

    
 


