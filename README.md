# molrec
recsys for ligands
-----


Predicting ligands using recommender system algorithms.

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
