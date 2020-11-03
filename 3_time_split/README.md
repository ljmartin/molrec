# time_split

comparing recommender systems and label correlation to the nearest neighbor approach on a time-split hold out set

to compare the recommender systems, run:

```
python compare_ranks.py
```

the ranks for the hold out data will be saved in `processed_data`.

Next, we wish to compare the performance of the structure-blind approaches to a nearest-neighbor network-based approach.
To do this we need molecular fingerprints. Run:

```
python make_fingerprints.py
```

the 2048-bit fingerprints get saved as a sparse matrix (`morgan.npz`).