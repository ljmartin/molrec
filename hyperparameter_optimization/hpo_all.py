#normal tools:
from scipy import sparse
import numpy as np
import copy
import sys
sys.path.append("..")
import utils

#learning library:
import lightfm
import implicit

#skopt:
from skopt.space import Real, Integer
from skopt import Optimizer
from sklearn.externals.joblib import Parallel, delayed
from skopt.utils import use_named_args
import skopt



#this performs multiple repeats of the test/train split, if desired:
def bootstrap(params, interaction_matrix, algorithm_function, repeats):
    results = list()
    for _ in range(repeats):
        #get a train/test split:
        train, test = utils.train_test_split(interaction_matrix, 0.05)
        #train matrix is used to train the model and make predictions:
        pred_matrix = algorithm_function(params,train)
        #test matrix is used to score the predictions:
        results.append(utils.evaluate_predictions(pred_matrix, test, train).mean())                
    return np.mean(results)


####SKOPT:
def run_skopt(algorithm_function, space, interaction_matrix):
    #the objective function for skopt:
    @use_named_args(space)
    def score(**params):        
        score = bootstrap(params, interaction_matrix, algorithm_function, 3)
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

    return result

def write_results(filename, result, space):
    #write the data so we don't need to repeat it:
    outfile = open(filename, 'w')

    outfile.write('Best parameters:\n')
    for j,k in zip(space, result.x_iters[np.argmin(result.func_vals)]):
        outfile.write(str(j.name)+' '+str(k)+'\n')
    outfile.write('Result: ')
    outfile.write(str(result.fun))

        
    outfile.write('\n\nAll:\n')
    for j,k in zip(result.x_iters, result.func_vals):
        outfile.write(str(j)+' '+str(k)+'\n')
    outfile.close()



if __name__ == '__main__':

    ##load the interaction matrix data for the subset of 243 proteins:
    interaction_matrix = utils.load_subset()

    
    ##Set the order of algorithms to be tested:
    algorithms = [utils.train_implicit_als,
                  utils.train_implicit_bpr,                  
                  utils.train_lightfm_warp,
                  utils.train_lightfm_bpr]

#Log has been removed because it requires negative records (ChEMBL highly biased to positive)
#utils.train_implicit_log,                  
#utils.train_lightfm_log]
    

    ##each algo has it's own parameter names and search space
    ##(lightfm's are all the same but it's easier just having multiple copies)
    spaces = list()
    #implicit,als:
    spaces.append([Integer(1, 400, name='factors'),
        Real(10**-5, 10**0, "log-uniform", name='regularization'),
        Integer(1,20, name='iterations')])
    #implicit,bpr:
    spaces.append([Integer(1, 400, name='factors'),
        Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
        Real(10**-5, 10**0, "log-uniform", name='regularization'), 
        Integer(1,20, name='iterations')])
##Log has been removed:
#    #implicit,log:
#    spaces.append([Integer(1, 400, name='factors'),
#        Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
#        Real(10**-5, 10**0, "log-uniform", name='regularization'), 
#        Integer(1,20, name='iterations')])
    #lightfm,warp:
    spaces.append([Integer(1, 400, name='no_components'),
        Integer(1,15, name='max_sampled'),
        Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
        Integer(1,20, name='epochs')])
    #lightfm,bpr:
    spaces.append([Integer(1, 400, name='no_components'),
        Integer(1,15, name='max_sampled'),
        Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
        Integer(1,20, name='epochs')])
##Log has been removed:
#    #lightfm,log:
#    spaces.append([Integer(1, 400, name='no_components'),
#        Integer(1,15, name='max_sampled'),
#        Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
#        Integer(1,20, name='epochs')])

    
    ##associated filenames for the outputs:
    names = ['hpo_implicit_als.dat', 'hpo_implicit_bpr.dat',
             'hpo_lightfm_warp.dat', 'hpo_lightfm_bpr.dat']

#    names = ['hpo_implicit_als.dat', 'hpo_implicit_bpr.dat', 'hpo_implicit_log.dat',
#             'hpo_lightfm_warp.dat', 'hpo_lightfm_bpr.dat', 'hpo_lightfm_log.dat']


    ##Run through each algo, send the 'space' to skop, run HPO, and write output file:
    for algo, space, name in zip(algorithms, spaces, names):
        result = run_skopt(algo, space, interaction_matrix)
        write_results(name, result, space)
