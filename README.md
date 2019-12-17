# molrec
##### recsys for ligands
---


Predicting ligands using recommender system algorithms.

Structure-based and ligand-based virtual screening tools have seen some wins but are either very [brute force](https://www.nature.com/articles/s41586-019-0917-9) or highly [baised to existing structures](https://pubs.acs.org/doi/10.1021/acs.jcim.7b00403). Ideally, a virtual screening technique can succesfully predict new ligands both efficiently and with non-obvious scaffolds. That means the predicted ligands must not arise by straight up similarity search from the known ligands, which we will use as a stand-in for an chemist's recommendation.

This project explores the use of network-based algorithms for this task. Data is activity records from ChEMBL25. Hyperparameter optimization uses a 243-target subset of ChEMBL and k-fold bootstrapping (structure bias is impossible for network-based algorithms, which are not fed any ligand or protein structure information). The k-fold bootstrapping procedure is required because of the requirement for network-based algorithms to have at least a single interaction to learn from in the train matrix - this is not gauranteed in k fold cross validation. 

The algorithms are then compared using a single time-split. 

The best-performing algorithm is then used for predicting either: 
* New ligands, for a target of interest, that are close-neighbours to an approved drug (to reduce clinical trials)
* New ligands, for a target of interest, that are _not_ close-neighbours known ligands (i.e. to discover new scaffold) 

To do:
- [x] write readme - complete 16-12-19
- [x] upload small dataset and parsing script - complete 16-12-19
- [x] create a common assessment task to compare any method on the 243-target subset. This will be the objective function for HPO. - complete 16-12-19
- [x] `implicit` hyperparameter optimization
- [ ] `lightfm` hyperparameter optimization
- [ ] `surprise` hyperparameter optimization
- [ ] [optional] [keras hyperparameter optimization](https://www.onceupondata.com/2019/02/10/nn-collaborative-filtering/)
- [ ] compare `label correlation`, `implicit-bpr`, `implicit-als`, and `lightfm-warp`, `lightfm-bpr` algorithms using time-split. Must output figures and data. [optional] keras 
- [ ] calculate number of known negatives (if any) predicted by best technique from above
- [ ] upload large dataset and parsing script
- [ ] determine top predictions for target of interest
- [ ] determine nearest approved molecules
- [ ] compare nearest approved molecules to those predicted by similarity search with known ligands
- [ ] alternative to approved drugs --> optimize both rank and Tanimoto distance to known ligands
