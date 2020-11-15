import matplotlib.pyplot as plt
from scipy.special import logit, expit
import numpy as np
plt.style.use('seaborn')
import sys
sys.path.append("..")
import utils

def plot_fig_label(ax, lab):
    ax.text(0, 1.15, lab, transform=ax.transAxes,
        fontsize=24, va='top', ha='left')


filenames = ['label_correlation', 'hpo_implicit_als', 'hpo_implicit_bpr',
             'hpo_lightfm_warp', 'hpo_lightfm_warp_fp', 'hpo_lightfm_bpr']


if __name__=='__main__':
    yrs = [2010, 2011, 2012, 2013, 2014, 2015, 2016]
    num_targets = []
    for yr in yrs:
        train, test = utils.load_time_split(year=yr, return_fingerprints=False)
        num_targets.append(train.shape[1])

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
        a.set_ylim(1, 50)
        a.set_title(name)
    fig.savefig('./figures/supp.pdf')
    fig.savefig('./figures/supp.tif')
    fig.savefig('./figures/supp.svg')
