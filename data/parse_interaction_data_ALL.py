import pandas as pd
import numpy as np
from scipy import sparse
import copy
from tqdm import tqdm

df = pd.read_csv('./interaction_data_ALL_species.csv')
df.columns = ['chembl_id', 'pref_name', 'instance_id', 'canonical_smiles']

#Remove duplicates and ensure each target has >=150 ligands:
df = df.drop_duplicates(['pref_name', 'instance_id'],keep='first').sort_values('pref_name')
keeplist = list()
for i in df['pref_name'].value_counts().items():
    if i[1]>=150:
        keeplist.append(i[0])

df = df[df['pref_name'].isin(keeplist)]


#Make into a n,m matrix:
num_instances = df['instance_id'].unique().shape[0]
num_targets = df['pref_name'].unique().shape[0]

interaction_matrix = np.zeros([num_instances, num_targets])


##Setting up containers:
tids = df.sort_values('pref_name')['pref_name'].unique()
cids = df.sort_values('instance_id')['instance_id'].unique()
target_indices = dict()
for count, i in enumerate(tids):
    target_indices[i]=count
    
instance_indices = dict()
for count, i in enumerate(cids):
    instance_indices[i]=count

    
##Filling the values:
for count, item in tqdm(df.iterrows(), total=len(df)):
    t_id = item['pref_name']
    i_id = item['instance_id']
    
    row = instance_indices[i_id]
    column = target_indices[t_id]
    
    interaction_matrix[row, column]=1

#test to make sure nothing went wrong:
for _ in range(100):
    row = np.random.choice(interaction_matrix.shape[0]) #select a random instance
    col = np.random.choice(interaction_matrix[row].nonzero()[0]) #select a label for that instance
    assert tids[col] in list(df[df['instance_id']==cids[row]]['pref_name'])
    
print('passed')
print(interaction_matrix.shape)


sparse.save_npz('./interaction_matrix_ALL_species.npz', sparse.csr_matrix(interaction_matrix))

df.sort_values('instance_id').drop_duplicates(['instance_id'])[['instance_id', 'canonical_smiles']].to_csv('all_chemicals')
df.sort_values('pref_name').drop_duplicates(['pref_name'])['pref_name'].to_csv('all_targets', index=False, header=None)

