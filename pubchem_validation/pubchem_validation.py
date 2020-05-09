import pandas as pd
from pprint import pprint
import time as time
import signal
import re

from tqdm import tqdm

import scipy
from scipy import sparse
from scipy import stats

import pubchempy as pcp
from chembl_webresource_client.new_client import new_client 
import json
import requests

import copy

import sys
sys.path.append("..")
import utils
import numpy as np



#all labels:
interaction_matrix = sparse.load_npz('../data/interaction_matrix_pchembl.npz')
smiles = pd.read_csv('../data/pchembl_chemicals.csv')
targets = pd.read_csv('../data/subset_targets.csv')
probability_matrix = utils.train_label_correlation(interaction_matrix)
probability_arr = probability_matrix.toarray()

arr = probability_matrix.toarray()
arr = arr - interaction_matrix
arr_sorted = np.dstack(np.unravel_index(np.argsort(-arr.ravel()), (arr.shape[0], arr.shape[1])))[0]
print('Should be a high number < 1:')
print(probability_arr[arr_sorted[0][0]][arr_sorted[0][1]])
print('Should be a low number >= 0:')
print(probability_arr[arr_sorted[-1][0]][arr_sorted[-1][1]])
print('Sorted array indices:')



def clean_text(input_string):
    #source: https://stackoverflow.com/questions/34860982/replace-the-punctuation-with-whitespace
    #replace these with whitespace:
    clean_string = re.sub(r"""
               [(),.;@#?!&$]+  # Accept one or more copies of punctuation
               \ *           # plus zero or more copies of a space,
               """,
               " ",          # and replace it with a single space
               input_string.lower(), flags=re.VERBOSE)
    
    #replace these with nothing:
    clean_string = clean_string.replace('-', '')
    clean_string = clean_string.replace('=', '')
    return clean_string


def get_synonyms(tid):
    target = new_client.target
    res = target.filter(target_chembl_id=tid)
    target_synonyms = [i['component_synonym'] for i in res[0]['target_components'][0]['target_component_synonyms']]
    #clean:
    target_synonyms = [clean_text(i) for i in target_synonyms]
    #make all lowercase to improve correct matchings:
    #target_synonyms = [i.lower() for i in target_synonyms]
    #remove all punctuations to improve correct matchings:
    #target_synonyms = [i.translate(str.maketrans('', '', string.punctuation)) for i in target_synonyms]
    
    return target_synonyms

def get_cid(smi):
    try:
        c = pcp.get_compounds(smi, 'smiles')[0]
        return c.cid
    except Exception as e:
        print(e)
        return None

def get_assay_summary(cid):
    b = json.loads(requests.get('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/'+str(cid)+'/assaysummary/JSON').content)
    return b

def get_relevant_aids(assay_summary, synonyms):
    #iterates through all assays and checks for overlap in Assay Name with any of the synonyms. 
    #if there is a shared word, returns the pubchem assay ID. 
    #relevant_aids = list()
    bioactivity_outcomes = list()
    assay_names = list()
    
    for i in assay_summary['Table']['Row']:
        assay_name = i['Cell'][11]
        #trick from word embedding - remove all punctuations to improve word matching
        #assay_name = assay_name.translate(str.maketrans('', '', string.punctuation))
        clean_assay_name = clean_text(assay_name)
        #now match words:
        if len(set(synonyms).intersection(clean_assay_name.split()))>0:
            
            ###This is the variable that stores the 'active' or 'unspecified' or 'inactive' string:
            bioactivity_outcome = i['Cell'][6]
            ###
            
            bioactivity_outcomes.append(bioactivity_outcome)
            assay_names.append(assay_name)
            
            #this stores the AID number
            #relevant_aids.append(i['Cell'][0])

    return bioactivity_outcomes, assay_names#relevant_aids

def get_assay_details(aid, cid):
    b = json.loads(requests.get('https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/'+str(aid)+'/JSON?cid='+str(cid)).content)
    return b

def get_pair_details(pair):
    smi = smiles['canonical_smiles'].iloc[pair[0]]
    instance_id = smiles['instance_id'].iloc[pair[0]]
    predicted_target = targets['pref_name'].iloc[pair[1]]
    tid = targets[targets['pref_name']==predicted_target]['chembl_id'].iloc[0]
    return smi, instance_id, tid, predicted_target


def fetch_assay_details(tid, smi):
    if tid in synonym_dict:
        synonyms = synonym_dict[tid]
    else:
        synonyms = get_synonyms(tid)
        synonym_dict[tid] = synonyms

        
    if smi in cid_dict:
        compound_id = cid_dict[smi]
    else:
        compound_id = get_cid(smi)
        cid_dict[smi] = compound_id
        
    if compound_id in assay_dict:
        assay_summary = assay_dict[compound_id]
    else: 
        assay_summary = get_assay_summary(compound_id)
        assay_dict[compound_id]=assay_summary
        
    return synonyms, compound_id, assay_summary

count = 0 
synonym_dict = dict()
cid_dict = dict()
assay_dict = dict()
assays_long = pd.DataFrame(columns=['ligandIdx', 'targetIdx', 'instance_id', 'pref_name', 'outcome', 'assayname'])
rownum=0

###This handles annoying cases that take forever (i.e. hung process)
#Close session after 15 seconds:
def handler(signum, frame):
    print('Time alarm')
    raise Exception('Action took too much time')
def signal_handler(signal, frame):
    print("\nprogram exiting gracefully")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler) #


for count, pair in tqdm(enumerate(arr_sorted[2595:10000]), smoothing=0, total=10000):
    print(f'testing {count}th pair: {pair} ... ', end=' ')
    #if the try block takes more than 15 seconds, kill it.
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(15) #Set the parameter to the amount of seconds you want to wait
        
    try:
        smi, instance_id, tid, pref_name = get_pair_details(pair)
    
        synonyms, compound_id, assay_summary = fetch_assay_details(tid, smi )
    
    
        if 'Fault' in assay_summary.keys():
            None
            #print('No assays present.')
        else:
            bioactivity_outcomes, assay_names = get_relevant_aids(assay_summary, synonyms)
            for outcome, aname in zip(bioactivity_outcomes, assay_names):
                assays_long.loc[rownum]=[pair[0], pair[1], instance_id, pref_name, outcome, aname]
                rownum += 1
    
        if count%100==0:
            assays_long.to_csv('assays_long.csv')
            with open('synonym_dict.json', 'w') as fp:
                json.dump(synonym_dict, fp)
            with open('assay_dict.json', 'w') as fp:
                json.dump(assay_dict, fp)
            with open('cid_dict.json', 'w') as fp:
                json.dump(cid_dict, fp)
        print(' - finished.')
    except (KeyboardInterrupt, Exception):
        print('took too long. moving on.')

assays_long.to_csv('assays_long.csv')
with open('synonym_dict.json', 'w') as fp:
    json.dump(synonym_dict, fp)
with open('assay_dict.json', 'w') as fp:
    json.dump(assay_dict, fp)
with open('cid_dict.json', 'w') as fp:
    json.dump(cid_dict, fp)
