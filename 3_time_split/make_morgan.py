from scipy import sparse
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors

import tqdm

#def makeMols(num=None):
#    
#    if num != None:
#        smiles = smiles[:num]
#    mols = list()
#    for smile in tqdm.tqdm(smiles):
#        mols.append(Chem.MolFromSmiles(smile))
#    return np.array(mols)

def get_morgan(smiles):
    fingerprint_function = rdMolDescriptors.GetMorganFingerprintAsBitVect
    pars = { "radius": 2,
             "nBits": 32768,
             "invariants": [],
             "fromAtoms": [],
             "useChirality": False,
             "useBondTypes": True,
             "useFeatures": False,
            }
    #store bit indices in these:
    row_idx = list()
    col_idx = list()
    #iterate through mols, 
    for count, smi in tqdm.tqdm(enumerate(smiles), total=len(smiles), smoothing=0):
        mol = Chem.MolFromSmiles(smi)
        fp = fingerprint_function(mol, **pars)
        onbits = list(fp.GetOnBits())
        #these bits all have the same row:
        row_idx += [count]*len(onbits)
        #and the column indices of those bits:
        col_idx+=onbits

    unfolded_size = 32768
    fingerprint_matrix = sparse.coo_matrix((np.ones(len(row_idx)).astype(bool), (row_idx, col_idx)), 
                          shape=(max(row_idx)+1, unfolded_size))
    #convert to csr matrix, it is better:
    fingerprint_matrix =  sparse.csr_matrix(fingerprint_matrix).astype('int')
    return fingerprint_matrix

if __name__ == '__main__':

    
    #mols = makeMols()
    smiles = pd.read_csv('../0_data/pchembl_chemicals.csv')['canonical_smiles']
    fps = get_morgan(smiles)
    sparse.save_npz('./morgan.npz', fps)
