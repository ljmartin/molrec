import numpy as np
import re
import pubchempy as pcp
from chembl_webresource_client.new_client import new_client 
import json
import requests
import sys
sys.path.append("..")
import utils

from scipy import sparse


##The following is to calculate AVE bias:
def fast_jaccard(X, Y=None):
    """credit: https://stackoverflow.com/questions/32805916/compute-jaccard-distances-on-sparse-matrix"""
    if isinstance(X, np.ndarray):
        X = sparse.csr_matrix(X)
    if Y is None:
        Y = X
    else:
        if isinstance(Y, np.ndarray):
            Y = sparse.csr_matrix(Y)
    assert X.shape[1] == Y.shape[1]

    X = X.astype(bool).astype(int)
    Y = Y.astype(bool).astype(int)
    intersect = X.dot(Y.T)
    x_sum = X.sum(axis=1).A1
    y_sum = Y.sum(axis=1).A1
    xx, yy = np.meshgrid(x_sum, y_sum)
    union = ((xx + yy).T - intersect)
    return (1 - intersect / union).A


class PubChemValidator(object):
    def __init__(self, targets_df, interaction_matrix, fps):
        self.tdf = targets_df
        self.interaction_matrix = interaction_matrix
        self.fps = fps

        self.ligands = {}
        self.targets = {}
        self.predictions = {}
    
    def load_checkpoint(self):
        self.ligands = json.load(open('ligands.json', 'r'))
        self.targets = json.load(open('targets.json', 'r'))
        self.predictions = json.load(open('predictions.json', 'r'))
        
    def save_checkpoint(self):
        with open('ligands.json', 'w') as fp:
            json.dump(self.ligands, fp)
        with open('targets.json', 'w') as fp:
            json.dump(self.targets, fp)
        with open('predictions.json', 'w') as fp:
            json.dump(self.predictions, fp)
            
    def has_ligand(self, idx):
        return str(idx) in self.ligands
    
    def has_target(self, idx):
        return str(idx) in self.targets

    def has_prediction(self, l_idx, t_idx):
        return str(l_idx)+':'+str(t_idx) in self.predictions
    
    def create_prediction(self, l_idx, t_idx, prob):
        record = dict()
        record['prob'] = str(prob)
        nn = self.get_nnrank_of_target(l_idx, t_idx)
        record['nn'] = nn
        
        self.predictions[str(l_idx)+':'+str(t_idx)] = record
        
    def create_target(self, idx):
        self.targets[str(idx)] = dict()
        record = self.targets[str(idx)]
    
        pref_name = self.tdf['pref_name'].iloc[idx]
        tid = self.tdf[self.tdf['pref_name']==pref_name]['chembl_id'].iloc[0]
        synonyms = get_synonyms(tid)
    
        record['pref_name'] = pref_name
        record['tid'] = tid
        record['synonyms'] = synonyms
        
    def create_ligand(self, idx):
        self.ligands[str(idx)] = dict()
        record = self.ligands[str(idx)]
    
        smi = smiles['canonical_smiles'].iloc[idx]
        chemblid = smiles['instance_id'].iloc[idx]
        cid = self.get_cid(smi)
        assays = self.get_assay_summary(cid)
        assays_parsed = self.parse_assays(assays)
    
        record['smi']=smi
        record['chemblid'] = chemblid
        record['cid'] = cid
        record['assays'] = assays_parsed
        
    def get_cid(self, smi):
        try:
            c = pcp.get_compounds(smi, 'smiles')[0]
            return c.cid
        except Exception as e:
            print(e)
            return 'cid_failed'
        
    def get_synonyms(self, tid):
        target = new_client.target
        res = target.filter(target_chembl_id=tid)
        synonyms = [i['component_synonym'] for i in res[0]['target_components'][0]['target_component_synonyms']]
        #clean:
        synonyms = [self.clean_text(i) for i in target_synonyms]
        return synonyms
    
    def clean_text(self, input_string):
        #source: https://stackoverflow.com/questions/34860982/replace-the-punctuation-with-whitespace
        #replace these with whitespace:
        clean_string = re.sub(r"""
               [(),.;@#?!&$]+  # Accept one or more copies of punctuation
               \ *           # plus zero or more copies of a space,
               """,
               " ",          # and replace it with a single space
               input_string.lower(), flags=re.VERBOSE)
    
        #replace these with nothing:
        clean_string = clean_string.replace('-', ' ')
        clean_string = clean_string.replace('=', '')
        return clean_string

    def get_assay_summary(self, cid):
        b = json.loads(requests.get('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/'+str(cid)+'/assaysummary/JSON').content)
        return b
    
    def parse_assays(self, assays):
        assays_parsed = []
        for assay in assays['Table']['Row']:
            cell = assay['Cell']
            aid = cell[0]
            name = self.clean_text(cell[11])
            activity = cell[6]
            
            assays_parsed.append([aid, activity, name])
        return assays_parsed
        
    def get_nnrank_of_target(self, ligand_idx, target_idx):

        positives = self.interaction_matrix[ligand_idx].nonzero()[1]
        all_distances = fast_jaccard(self.fps[ligand_idx], self.fps)[0]
        s = np.argsort(all_distances)

        pred = target_idx
        curr_rank = 0
        count=1
        preds = []
        seen = []

        while pred not in seen:
            predictions = self.interaction_matrix[s[count]].nonzero()[1]
            preds = np.setdiff1d(predictions,positives)
            preds = np.setdiff1d(preds, seen)
            curr_rank = len(seen)
            seen += list(preds)
            if len(preds)>0:
                 curr_rank+= np.mean(np.arange(len(preds))+1)
            count+=1

        return curr_rank
