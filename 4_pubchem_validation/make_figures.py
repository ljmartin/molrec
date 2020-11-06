import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import altair as alt
import tqdm


targets_df= pd.read_csv('../0_data/subset_targets.csv')
pcv = PubChemValidator(None, None, None)
pcv.load_checkpoint()


df = pd.DataFrame(columns=['tid', 'pref_name', 'lig_chemblid', 'aid', 'outcome', 'title','nnrank'])
count = 0 
for pred in tqdm.tqdm_notebook(pcv.predictions.keys()):
    lig, targ = pred.split(':')
    try:
        target_synonyms = pcv.targets[targ]['synonyms']
        tid = pcv.targets[targ]['tid']
        pref_name = pcv.targets[targ]['pref_name']        
        ligand_chemblid = pcv.ligands[lig]['chemblid']
        assays = pcv.ligands[lig]['assays']
    except:
        continue
    
    nn = pcv.predictions[pred]['nn']   
    for a in assays:
        aid = a[0]
        title = a[2]
        clean_title = pcv.clean_text(title)

        #see if the target_synonym is mentioned in the title:
        num = len(set(target_synonyms).intersection(clean_title.split()))
        #if so, then:
        if num>0:
            outcome =  a[1]
            if outcome in ['Active', 'Inactive']:
                df.loc[count] = [tid, pref_name, ligand_chemblid, aid, outcome, title, nn]
                count+=1
