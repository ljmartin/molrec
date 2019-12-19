#normal tools:
from scipy import sparse
import numpy as np
import copy
import sys
sys.path.append("..")
import utils

#learning libraries:
import implicit
import lightfm

#skopt:
from skopt.space import Real, Integer
from skopt import Optimizer
from sklearn.externals.joblib import Parallel, delayed
from skopt.utils import use_named_args
import skopt


#load the 243-protein subset:
interaction_matrix = utils.load_subset()
