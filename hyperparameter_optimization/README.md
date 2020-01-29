# Hyperparameter optimization for `implicit` and `lightfm`
Uses scikit-optimize to find close-to-optimum hyperparameters for running `implicit` and `lightfm` models. There are 20 random search iterations, after which a tree-based regression model (from sklearn) is fit to find the next-best point. As seen in the figure, logistic matrix factorization does not really find a solution in either case. For BPR, ALS, and WARP, 20 random search points seems sufficient, and the tree-based optimization converges quickly afterwards.  


To run analysis, simple as:

```
bash run_all.sh
```