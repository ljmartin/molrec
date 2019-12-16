# molrec
recsys for ligands


Predicting ligands using recommender system algorithms.

Structure-based and ligand-based virtual screening tools have seen some wins but are either [very brute force](https://www.nature.com/articles/s41586-019-0917-9) or highly [baised to existing structures](https://pubs.acs.org/doi/10.1021/acs.jcim.7b00403). Ideally, a virtual screening technique can succesfully predict new ligands with 

To do:
- [ ] write readme
- [ ] upload datasets
- [ ] create a common assessment task to compare any method on the 243-target subset. Must output data and/or figures.
- [ ] `implicit` hyperparameter optimization
- [ ] `lightfm` hyperparameter optimization
- [ ] [optional] [keras hyperparameter optimization](https://www.onceupondata.com/2019/02/10/nn-collaborative-filtering/)
- [ ] compare `label correlation`, `implicit-bpr`, `implicit-als`, and `lightfm-warp`, `lightfm-bpr` algorithms. [optional] keras 
- [ ] calculate number of known negatives (if any) predicted by best technique from above
- [ ] determine top predictions for target of interest
- [ ] determine nearest approved molecules
- [ ] compare nearest approved molecules to those predicted by similarity search with known ligands
- [ ] alternative to approved drugs --> optimize both rank and Tanimoto distance to known ligands
