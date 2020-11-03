# How to make data:

To do this from the very beginning, you must have a local copy of the chembl_25.db database, which is a large file that doesn't fit here. It is freely available from chembl.  

Code to setup the data from the beginning (you can skip this if you just wanted to use the sparse matrices and smiles file included in the repo):

```
sqlite3 chembl_25.db < get_interaction_data_SUBSET_pchembl.sql
python parse_interaction_data_SUBSET.py pchembl
```

The `pchembl` input asks to parse the data where pchembl values of >=5 are defined as active. Alternatively one could use `active` to also include ChEMBL records with 'active' in their Activity Comment field - however these do not necessarily have an associated dose/response measurement that confirms the result is not a false positive as per recommended guidelines for PubChem assay data. 

This paper uses `pchembl`.
