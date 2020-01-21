#normal tools:
from scipy import sparse
import numpy as np
import copy
import sys
sys.path.append("..")
import utils

#learning library:
import lightfm

#skopt:
from skopt.space import Real, Integer
from skopt import Optimizer
from sklearn.externals.joblib import Parallel, delayed
from skopt.utils import use_named_args
import skopt




#load the 243-protein subset:
interaction_matrix = utils.load_subset()

##lightfm requires a particular way of getting the predictions:
#lightfm 'user id' (chemical id)
cid = np.arange(interaction_matrix.shape[0])
#lightfm 'item id' (target id)
tid = np.arange(interaction_matrix.shape[1])


#this performs multiple repeats of the test/train split, if desired:
def bootstrap(params, matrix, repeats):
    results = list()
    for _ in range(repeats):
        train, test = utils.train_test_split(interaction_matrix, 0.05)
        model = lightfm.LightFM(no_components = params['no_components'],
                           loss='logistic', 
                           max_sampled=params['max_sampled'],
                           learning_rate=params['learning_rate'])
        model.fit(train, epochs=params['epochs'])
        #make interaction predictions:
        pred_matrix = model.predict(np.repeat(cid, len(tid)), np.tile(tid, len(cid)))
        pred_matrix = np.reshape(pred_matrix, (len(cid), len(tid)))
        #evaluate by calculating mean rank:
        results.append(-utils.evaluate_predictions(pred_matrix, train, test))
                
    return np.mean(results)


####SKOPT:

#these are the hyperparameters and search spaces:
space = [Integer(1, 400, name='no_components'),
        Integer(1,15, name='max_sampled'),
        Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
        Integer(1,20, name='epochs')]

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
outfile = open('hpo_lightfm_log.dat', 'w')

outfile.write('Best:\n')
outfile.write('Paramaters: ')
outfile.write(str(result.x_iters[np.argmin(result.func_vals)]))
outfile.write('\nResult: ')
outfile.write(str(result.fun))


outfile.write('\n\nAll:\n')
for j,k in zip(result.x_iters, result.func_vals):
    outfile.write(str(j)+' '+str(k)+'\n')
outfile.close()
