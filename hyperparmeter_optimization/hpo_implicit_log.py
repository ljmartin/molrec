#normal tools:
from scipy import sparse
import numpy as np
import copy
import sys
sys.path.append("..")
import utils

#learning library:
import implicit

#skopt:
from skopt.space import Real, Integer
from skopt import Optimizer
from sklearn.externals.joblib import Parallel, delayed
from skopt.utils import use_named_args
import skopt




#load the 243-protein subset:
interaction_matrix = utils.load_subset()

#this performs multiple repeats of the test/train split, if desired:
def bootstrap(params, matrix, repeats):
    results = list()
    for _ in range(repeats):
        train, test = utils.train_test_split(interaction_matrix, 0.05)
        model = implicit.lmf.LogisticMatrixFactorization(factors=params['factors'],
                                                     learning_rate=params['learning_rate'],
                                                     regularization=params['regularization'],
                                                     iterations=params['iterations'], 
                                                     num_threads=1,
                                                     use_gpu=False)
        model.fit(train)
        #make interaction predictions:
        pred_matrix = np.dot(model.item_factors, model.user_factors.T)
        #evaluate by calculating mean rank:
        results.append(utils.evaluate_predictions(pred_matrix, train, test))
                
    return np.mean(results)


####SKOPT:

#these are the hyperparameters and search spaces:
space = [Integer(1, 400, name='factors'),
        Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
        Real(10**-5, 10**0, "log-uniform", name='regularization'), 
        Integer(1,20, name='iterations')]


#the objective function for skopt:
@use_named_args(space)
def score(**params):        
    score = bootstrap(params, interaction_matrix, 1)
    return (score)

optimizer = Optimizer(dimensions=space, 
                     random_state=1,
                     base_estimator='ET',
                     n_random_starts=20)

#6*6 = 36 iterations, with 20 being random. 
for i in range(6):
    print(i)
    x = optimizer.ask(n_points=6)
    y = Parallel(n_jobs=6)(delayed(score)(v) for v in x)
    optimizer.tell(x,y)

result = skopt.utils.create_result(optimizer.Xi,
                                  optimizer.yi,
                                  optimizer.space,
                                  optimizer.rng,
                                  models=optimizer.models)



#write the data so we don't need to repeat it:
outfile = open('hpo_implicit_log.dat', 'w')

outfile.write('Best:\n')
outfile.write('Paramaters: ')
outfile.write(str(result.x_iters[np.argmin(result.func_vals)]))
outfile.write('\nResult: ')
outfile.write(str(result.fun))


outfile.write('\n\nAll:\n')
for j,k in zip(result.x_iters, result.func_vals):
    outfile.write(str(j)+' '+str(k)+'\n')
outfile.close()
