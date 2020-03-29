import numpy as np

from scipy import sparse

import pandas as pd
import six

import sys
sys.path.append("..")
import utils

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

