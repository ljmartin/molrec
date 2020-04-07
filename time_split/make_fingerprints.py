from scipy import sparse
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

import tqdm

def makeMols(num=None):
    smiles = pd.read_csv('../data/pchembl_chemicals.csv')['canonical_smiles']
    if num != None:
        smiles = smiles[:num]
    mols = list()
    for smile in tqdm.tqdm(smiles):
        mols.append(Chem.MolFromSmiles(smile))
    return np.array(mols)

def get_morgan(mols):
    gen_mo = rdFingerprintGenerator.GetMorganGenerator()
    fps = list()
    for mol in tqdm.tqdm(mols):
        try:
            fp = np.array(gen_mo.GetFingerprint(mol))
        except:
            fp = np.zeros(2048).astype(int)
        fps.append(fp)
    fps = np.array(fps)
    return sparse.csr_matrix(fps).astype('int')

if __name__ == '__main__':

    mols = makeMols()
    fps = get_morgan(mols)
    sparse.save_npz('./morgan.npz', fps)
