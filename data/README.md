# How to make data:

To do this from the very beginning, you must have a local copy of the chembl_25.db database, which is a large file that doesn't fit here. It is freely available from chembl.  

Code to setup the data from the beginning (you can skip this if you just wanted to use the sparse matrices and smiles file included in the repo):

```
sqlite3 chembl_25.db < .read get_interaction_data_SUBSET.sql
python parse_interaction_data_SUBSET.py
```
