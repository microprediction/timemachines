from timemachines.optimization import optimize
from timemachines.synthetic import brownian_with_exogenous
from timemachines.optimizers.alloptimizers import OPTIMIZERS
from timemachines.skaters.allskaters import SKATERS
import time
import random


def optimize_random_skater():
    start_time = time.time()
    optimizer = random.choice(OPTIMIZERS)
    f = random.choice(SKATERS)
    print('Running '+str(optimizer.__name__))
    r_star = optimize(f=f, ys=brownian_with_exogenous(n=150),
                      n_trials=15, optimizer=optimizer)
    print("Best hyper-param is " + str(r_star))
    print('Took ' + str( (time.time()-start_time)/60 ) + ' minutes.' )



if __name__=='__main__':
    optimize_random_skater()