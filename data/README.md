# How to make data:

Must have a local copy of the chembl_25.db database, which is a large file that doesn't fit here. It is freely available from chembl.  

Code to setup the data:

```
sqlite3 chembl_25.db < .read get_interaction_data_SUBSET.sql
python parse_interaction_data_SUBSET.py
```