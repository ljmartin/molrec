import pandas as pd
import numpy as np
from scipy import sparse
import copy
from tqdm import tqdm
import sys

fname = sys.argv[1]
df = pd.read_csv('interaction_data_SUBSET_'+fname+'.csv')
df.columns = ['chembl_id', 'organism', 'pref_name', 'instance_id', 'pchembl_value', 'canonical_smiles', 'year']


##
##Sorting by year puts the year=NaN last. This means the NaN-year entries will be removed if there is a duplicate with a real year. The dates of year=NaN year entries are ambiguous, so they cannot take part in time-split validation. They will always be in the training set.
df = df.sort_values('year').drop_duplicates(['pref_name', 'instance_id'],keep='first').sort_values('pref_name')

#Two of the pref_names in here are associated with multiple (2 each) ChEMBLIDs. Removing the smaller sized record in each case (because the original dataset has >250 interactions in each case!):
df = df[~(df['chembl_id']=='CHEMBL2111414')]
df = df[~(df['chembl_id']=='CHEMBL2111353')]

#NaN values will mess with numpy later so should be dealt with now. Setting all these to year=1 is functionally equivalent to ensuring these interactions are always the training set. #Ps - don't set to 0! Otherwise the '0' dates will be removed whenever the matrix becomes a sparse csr_matrix.
df = df.replace(np.nan, 1)

#Sklearn has a good function for creating a label matrix ("y"), but we want to have a duplicate matrix with the year of the interactions, so lets just do both at the same time.

#Setup some data objects:

num_instances = df['instance_id'].unique().shape[0]
num_targets = df['chembl_id'].unique().shape[0]

#interaction matrix:
interaction_matrix = np.zeros([num_instances, num_targets])
#interaction dates:
interaction_dates = copy.copy(interaction_matrix)

###setting up column indices, to use in filling in the matrices above
tids = df.sort_values('chembl_id')['chembl_id'].unique()
cids = df.sort_values('instance_id')['instance_id'].unique()
target_indices = dict()
for count, i in enumerate(tids):
    target_indices[i]=count

instance_indices = dict()
for count, i in enumerate(cids):
    instance_indices[i]=count


#Actually filling the values:

for count, item in tqdm(df.iterrows()):
    t_id = item['chembl_id']
    i_id = item['instance_id']
    date = item['year']

    row = instance_indices[i_id]
    column = target_indices[t_id]
    
    interaction_matrix[row, column] = 1
    interaction_dates[row, column] = date


#Do a little test to make sure some randomly chosen positives in the interaction_matrix line up with real entries in the df.

for _ in range(100):
    row = np.random.choice(interaction_matrix.shape[0]) #select random instance
    col = np.random.choice(interaction_matrix[row].nonzero()[0]) #select from positives of that instance
    assert tids[col] in list(df[df['instance_id']==cids[row]]['chembl_id'])
    
print('passed')
print('Matrix shape:', interaction_matrix.shape)

##Save all the data
sparse.save_npz('../data/interaction_dates_'+fname+'.npz', sparse.csr_matrix(interaction_dates))
sparse.save_npz('../data/interaction_matrix_'+fname+'.npz', sparse.csr_matrix(interaction_matrix))


df.sort_values('instance_id').drop_duplicates(['instance_id'])[['instance_id', 'canonical_smiles']].to_csv('../data/'+fname+'_chemicals.csv', index=False)
df.sort_values('chembl_id').drop_duplicates(['chembl_id'])['pref_name'].to_csv('subset_targets', index=False, header=None)
df.sort_values('chembl_id').drop_duplicates(['chembl_id']).to_csv('subset_targets.csv', index=False)
