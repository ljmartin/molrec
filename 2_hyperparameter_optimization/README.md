# Hyperparameter optimization for `implicit` and `lightfm`
Uses scikit-optimize to find close-to-optimum hyperparameters for running `implicit` and `lightfm` models. There are 20 random search iterations, after which a tree-based regression model (from sklearn) is fit to find the next-best point. As seen in the figure, logistic matrix factorization does not really find a solution in either case. For BPR, ALS, and WARP, 20 random search points seems sufficient, and the tree-based optimization converges quickly afterwards.  


To run analysis, simple as:

```
bash run_all.sh
```

|      | Algorithm    | Hyperparameter | Value      |
| ---: | :----------- | :------------- | ---------- |
|      | implicit_als | factors        | 36         |
|      | implicit_als | regularization | 0.00162069 |
|      | implicit_als | iterations     | 6          |
|      | implicit_bpr | factors        | 186        |
|      | implicit_bpr | learning_rate  | 0.18615    |
|    5 | implicit_bpr | regularization | 0.00511469 |
|    6 | implicit_bpr | iterations     | 16         |
|    7 | lightfm_warp | no_components  | 127        |
|    8 | lightfm_warp | max_sampled    | 9          |
|    9 | lightfm_warp | learning_rate  | 0.0561297  |
|   10 | lightfm_warp | epochs         | 6          |
|   11 | lightfm_bpr  | no_components  | 8          |
|   12 | lightfm_bpr  | max_sampled    | 14         |
|   13 | lightfm_bpr  | learning_rate  | 0.162629   |
|   14 | lightfm_bpr  | epochs         | 19         |
