# molrec
##### recsys for ligands
---


Predicting ligands using recommender system algorithms.

Structure-based and ligand-based virtual screening tools have seen some wins but are either very [brute force](https://www.nature.com/articles/s41586-019-0917-9) or highly [baised to existing structures](https://pubs.acs.org/doi/10.1021/acs.jcim.7b00403). Ideally, a virtual screening technique can succesfully predict new ligands both efficiently and with non-obvious scaffolds. That means the predicted ligands could not have arisen by straight up similarity search starting from known ligands, which we will use as a stand-in for a chemist's recommendation.

(Adding to the above -> many existing algorithms require binary true positive and true negative labels. Network analysis allows for positive, negative, and unknown. In fact ChEMBL, which is highly biased to positive-only records, is better suited to implicit data techniques like WARP) 

This project explores the use of network-based algorithms for this task. Data is activity records from ChEMBL25. Hyperparameter optimization uses a 243-target subset of ChEMBL and k-fold bootstrapping (structure bias is impossible for network-based algorithms, which are not fed any ligand or protein structure information). The k-fold bootstrapping procedure is required because of the requirement for network-based algorithms to have at least a single interaction to learn from in the train matrix - this is not gauranteed in k fold cross validation. 

The algorithms are then compared using a single time-split. 

The best-performing algorithm is then used for predicting either: 
* New ligands, for a target of interest, that are close-neighbours to an approved drug (to reduce clinical trials)
* New ligands, for a target of interest, that are _not_ close-neighbours known ligands (i.e. to discover new scaffold) 

To do:
- [x] write readme - complete 16-12-19
- [x] upload small dataset and parsing script - complete 16-12-19
- [x] create a common assessment task to compare any method on the 243-target subset. This will be the objective function for HPO. - complete 16-12-19
- [x] `implicit` hyperparameter optimization - complete 17-12-19
- [x] `lightfm` hyperparameter optimization - complete 17-12-19
- [ ] ~~`surprise` hyperparameter optimization~~
- [x] convert label correlation approach to use sparse matrices
- [x] show that optimizing for wide vs long is equivalent or not equivalent. complete 22-01-2020
  _It is better to do long_!
- [ ] ~~determine what sklearn ranking loss returns in the event of a zero-vector~~ Not relevant for mean rank or p@k. 
- [ ] ~~decide on the ultimate scoring function. p@k? mean rank of test labels? What about inspection bias?~~ Answer: p@k is useful more for the long-format input data (with axis=1 ranking). But this isn't how we want to use the results! Best to just keep mean rank, or compare the CDF. 
- [ ] compare `label correlation`, `implicit-bpr`, `implicit-als`, and `lightfm-warp`, `lightfm-bpr` algorithms using time-split. Must output figures and data.
- [ ] calculate number of known negatives (if any) predicted by best technique from above
- [ ] upload large dataset and parsing script
- [ ] determine top predictions for target of interest
- [ ] determine nearest approved molecules
- [ ] compare nearest approved molecules to those predicted by similarity search with known ligands
- [ ] alternative to approved drugs --> optimize both rank and Tanimoto distance to known ligands

This requires explicit interaction data, i.e. ligand affinity values. Harry?:
- [ ] [optional] [keras hyperparameter optimization](https://www.onceupondata.com/2019/02/10/nn-collaborative-filtering/)
