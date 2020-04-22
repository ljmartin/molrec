import numpy as np

from scipy import sparse
from scipy.stats import gaussian_kde

import pandas as pd
import six

import sys
sys.path.append("..")
import utils

import pymc3 as pm

import tqdm
import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use('seaborn')

from statsmodels.distributions.empirical_distribution import ECDF



names = pd.read_csv('../data/subset_targets', header=None)
names.columns=['Protein Name']

all_interactions = sparse.load_npz('../data/interaction_matrix_pchembl.npz').toarray()
##Remove ligands with only 1 label:
mask = np.sum(all_interactions, axis=1)
mask = mask>1
interaction_matrix = all_interactions[mask]
#tot_instances includes single labels
tot_instances = all_interactions.sum(0)

#count fractions of ligands with more than one label. 
fraction_multilabel = interaction_matrix.sum(0)/tot_instances
names['%']=np.around(fraction_multilabel*100, 2)
names =names.sort_values(by='%', ascending=False)


##Do empirical cumulative distribution function
ec = ECDF(np.sum(all_interactions, axis=1))


def plot_fig_label(ax, lab):
    ax.text(-0.1, 1.15, lab, transform=ax.transAxes,
        fontsize=24, fontweight='bold', va='top', ha='right')

def render_mpl_table(data, sizes, col_width=10.0, row_height=0.625, font_size=12,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
    ax.axis('off')
    
    mpl_table = ax.table(cellText=data.values, 
                         bbox=bbox, 
                         colLabels=data.columns, 
                         cellLoc='center', 
                         colWidths=sizes,
                         **kwargs)
    

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
            cell.set_text_props(wrap=True, fontstretch='ultra-condensed')
    return ax


fig = plt.figure(figsize=(13,9))
grid = plt.GridSpec(2, 2, wspace=0.1, hspace=0.3)
ax0=plt.subplot(grid[0, 0])
ax1=plt.subplot(grid[1, 0])
ax2=plt.subplot(grid[:, 1])



ec = ECDF(np.sum(all_interactions, axis=1))
ax0.plot(ec.x, ec.y*100)
ax0.set_xscale('log')
ax0.set_xlabel('% multi-label ligands',fontsize=16)
ax0.set_xticks([1,3,10,30,100])
ax0.set_xticklabels([1,3,10,30,100])
ax0.set_ylabel('Cumulative probability', fontsize=16)
plot_fig_label(ax0, 'A')


#ax0.set_xticks([1,2,3,4,5,10,20,30,40,50,100]);
#ax0.set_xticklabels([[1,2,3,4,5,10,20,30,40,50,100]])
ax1.scatter(tot_instances, names['%'], alpha=0.7, linewidth=1, s=100, edgecolor='k')
ax1.set_xlabel('Total ligands per protein', fontsize=16)
ax1.set_ylabel('% multi-label ligands',fontsize=16)
plot_fig_label(ax1, 'B')

ax2 = render_mpl_table(names.iloc[:25], np.array([2.3,0.5]), header_columns=0, col_width=2.0, ax=ax2)
ax2.text(0, 1.08, 'C', transform=ax2.transAxes,
    fontsize=24, fontweight='bold', va='top', ha='right')

plt.tight_layout()

fig.savefig('basic_stats.pdf')
plt.close(fig)


##calculate correlations:
L = sparse.lil_matrix((interaction_matrix.shape[1], interaction_matrix.shape[1]))

for idx in tqdm.tqdm(range(interaction_matrix.shape[0]), smoothing=0.1):
    row = interaction_matrix[idx]
    if row.sum()>1:
        for j,k in itertools.permutations(row.nonzero()[0], 2):
            L[j,k] += (1)/(tot_instances[k])

corr = L.toarray()

#get protein labels:
numpy_names =pd.read_csv('../data/subset_targets', header=None)[0].to_numpy()
#rank pairwise correlations
ranks = np.dstack(np.unravel_index(np.argsort((-corr).ravel()), (corr.shape[0], corr.shape[0])))[0]

for i in range(numpy_names.shape[0]):
    length=len(numpy_names[i])
    if length>29:
        numpy_names[i]=numpy_names[i][:26]+'...'


##make a df with the top correlations:
prot1 = list()
prot2 = list()
similarity = list()
for j in ranks[:200:1]:
    prot1.append(numpy_names[j[0]])
    prot2.append(numpy_names[j[1]])
    similarity.append(np.around(100*corr[j[0]][j[1]], 1))

simdf = pd.DataFrame()
simdf['Protein 1'] = prot1
simdf['Protein 2'] = prot2
simdf['%'] = similarity


fig, ax = plt.subplots(1,3, gridspec_kw={'width_ratios': [1.5, 1.5, 1.5]})
fig.set_figwidth(18)
fig.set_figheight(6)
plot_fig_label(ax[0], 'A')

imsh = ax[0].imshow(corr, cmap='Blues_r')
ax[0].grid()
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(imsh, cax=cax, orientation='vertical')



render_mpl_table(simdf.iloc[:10], np.array([1.5,1.5,0.5]),ax=ax[1], font_size=8)
render_mpl_table(simdf.iloc[10:20], np.array([1.5,1.5,0.5]),ax=ax[2], font_size=8)
ax[1].text(0.05, 1.08, 'B', transform=ax[1].transAxes,
    fontsize=24, fontweight='bold', va='top', ha='right')


fig.subplots_adjust(hspace=0.05, wspace=0.1)


fig.savefig('label_correlation.pdf')
plt.close(fig)





##Plot LOO analysis of label correlation:

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

def calc_kde(ranks, xs=np.linspace(1,244,244)):
    #kde:
    density = gaussian_kde(ranks)
    density.covariance_factor= lambda : 0.25
    density._compute_covariance()
    return density(xs)

rank_arr = np.load('rank_arr.npy')
mean_trace = calc_hpd(np.array(rank_arr), np.mean)
median_trace = calc_hpd(np.array(rank_arr), np.median)


def plot_fig_label(ax, lab):
    ax.text(-0.15, 1.12, lab, transform=ax.transAxes,
        fontsize=24, va='top', ha='left')

fig, ax = plt.subplots(nrows=3,ncols=1)
fig.set_figheight(10)
fig.set_figwidth(6)

label='Label correlation'
for count,trace,name in zip([0,1], [mean_trace, median_trace], ['mean rank', 'median rank']):
    m = np.mean(trace['a'])
    hpd = pm.hpd(trace['a'])
    print(m, hpd)
    xs = np.linspace(m-3,m+3,100)
    density = calc_kde(trace['a'], xs=xs)

    ax[0].errorbar(count, m, yerr = np.array([m-hpd[0], hpd[1]-m])[:,None],mfc='white',mew=2,
                           fmt='o', c='C0',linewidth=4, markersize=15, capsize=3, label=label)
    label=None

ax[0].set_xticks([0,1])
ax[0].set_xticklabels(['Mean\nrank', 'Median\nrank'])
ax[0].set_xlim(-0.5,1.5)
#ax[0].errorbar(-10,3, label='Label correlation')
ax[0].legend(frameon=True, fancybox=True, framealpha=1)
ax[0].set_ylabel('Rank', fontsize=14)
ax[0].set_ylim(0.1,7)
ax[0].set_title('Mean and median rank', fontsize=14)
plot_fig_label(ax[0], 'A')

n, x = np.histogram(np.array(rank_arr), bins = np.linspace(1,243,243))
#bin_centers = 0.5*(x[1:]+x[:-1])
#ax[1].plot(bin_centers,n,'-o', mfc='white', label='Label correlation')
#ax[1].plot(x[:-1],n,'-o', mfc='white', label='Label correlation')
ax[1].plot(x[:-1],n,'-o', mfc='white', mew=1, label='Label correlation')
#ax[1].scatter(bin_centers,n,label='Label correlation')
ax[1].set_xlim(0,20)
ax[1].set_xticks(np.arange(1,20))
ax[1].set_xlabel('Rank', fontsize=14)
ax[1].set_ylabel('Count density',fontsize=14)
ax[1].set_title('Histogram of predicted ranks, top 20',fontsize=14)
ax[1].legend(frameon=True, fancybox=True, framealpha=1)
plot_fig_label(ax[1], 'B')

ecdf = np.cumsum(n)/n.sum()
ax[2].plot([1]+list(x[:-1]),[0]+list(ecdf), '-o', mfc='white', mew=1, label='Label correlation')
ax[2].set_xlim(0.0,20)
ax[2].set_xticks(np.arange(1,20))
ax[2].plot([0,3],[ecdf[2],ecdf[2]],c='C0', linestyle=':', label='ECDF at rank 3')
ax[2].legend(frameon=True, fancybox=True, framealpha=1)
ax[2].set_ylabel('Cumulative\nnormalized density', fontsize=14)
ax[2].set_xlabel('Rank',fontsize=14)
ax[2].set_title('ECDF, top 20',fontsize=14)
plot_fig_label(ax[2], 'C')


plt.tight_layout()

plt.savefig('label_correlation_loo.pdf')





##Plot calibration:
score_arr = np.load('score_arr.npy')

x =np.linspace(1,0,100)
metric = list()
for cutoff in x:
    ranks = rank_arr[score_arr>=cutoff]
    n = len(ranks)
    val =(ranks<=1).sum()/n
    metric.append(val)

fig, ax = plt.subplots()
fig.set_figheight(7.5)
fig.set_figwidth(7.5)

ax.plot(x*100, np.array(metric)*100, '-o', mfc='white', mew=1, label='Label correlation')
ax.set_ylabel('Percentage labels ranked 1 (%)', fontsize=14)
ax.set_xlabel('Predicted probability of an interaction (%)', fontsize=14)
ax.plot([0,100],[0,100], label='Perfect calibration')
ax.legend()
fig.savefig('calibration.pdf')    
