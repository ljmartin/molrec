import pandas as pd
import numpy as np
from scipy import sparse
from tqdm import tqdm
from scipy.stats.mstats import rankdata

import sys
sys.path.append("..")
import utils

#Load up large interaction matrix (human, rat, mouse proteins with >150 interactions each)
interaction_matrix = sparse.load_npz('../data/interaction_matrix_ALL_species.npz')

#Get the index of the GPR55 column:
targetNames = pd.read_csv('../data/all_targets', header=None).to_numpy().reshape(1,-1)[0]
targetIndex = (targetNames=='G-protein coupled receptor 55').nonzero()[0][0]



##Fit top three algorithms on the interaction matrix:
#LightFM is slower than the other two. 
filenames = ['label_correlation', 'hpo_implicit_bpr', 'hpo_lightfm_warp']

algorithms = [utils.train_label_correlation, utils.train_implicit_bpr, utils.train_lightfm_warp]

preds_list = list()
ranks_list = list()
for name, algo in zip(filenames, algorithms):
    if name == 'label_correlation':
        preds = algo(interaction_matrix)
        preds_list.append(preds[:,targetIndex].toarray().flatten())
    else:
        params = utils.read_params(name)
        preds = algo(params, interaction_matrix)
        for _ in tqdm(range(7)):
            preds += algo(params, interaction_matrix)
        preds_list.append(preds[:,targetIndex])


##Next, for each set of predictions, get the ranking 
##of only the unknown ligands (i.e. mask and remove the 
##known true positives)

ranks_list = list()
#mask to remove the true positives:
true_pos = interaction_matrix[:,targetIndex].toarray().T[0]==1

for preds in tqdm(preds_list):
    ranks = rankdata(-preds[~true_pos])
    ranks_list.append(ranks)

##Finally, take the geometric average of the three sets of ranks:
##to get a consensus set of rank orders:
geo_ranks =np.power(ranks_list[0] * ranks_list[1] * ranks_list[2], 1/3)

#this now returns the indices that give you the top ranked ligands.
ranked_indices = geo_ranks.argsort()



##Save top results:
#load all chemical smiles:
allSmiles = pd.read_csv('../data/all_chemicals')
#get top 2000 predicted unknowns
predicted_smiles = allSmiles.iloc[~true_pos].iloc[ranked_indices[:2000]][['instance_id', 'canonical_smiles']]
#and keep the smiles for the true positives as well:
true_smiles = allSmiles.iloc[true_pos][['instance_id', 'canonical_smiles']]

predicted_smiles.to_csv('predicted_smiles.csv', header=0, index=False)
true_smiles.to_csv('true_smiles.csv', header=0, index=False)

#save the positive labels so we know why each ligand was recommended:
positive_labels = interaction_matrix.toarray()[~true_pos][ranked_indices[:2000]]
np.save('positive_labels.npy', positive_labels)

