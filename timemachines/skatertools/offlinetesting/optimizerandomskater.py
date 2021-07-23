from timemachines.skatertools.tuning.hyper import optimal_r
from timemachines.skatertools.data.synthetic import brownian_with_exogenous
from timemachines.skaters.allskaters import SKATERS_R1, SKATERS_R2, SKATERS_R3
import time
import random
from pprint import pprint
from timemachines.skaters.proph.prophparams import PROPHET_META

try:
    from humpday.optimizers.alloptimizers import OPTIMIZERS
except ImportError:
    raise ImportError('You need to pip install humpday')


def optimize_random_skater():
    print('Available optimizers...')
    print([o.__name__ for o in OPTIMIZERS])
    start_time = time.time()
    optimizer = random.choice(OPTIMIZERS)
    f = random.choice(SKATERS_R1+SKATERS_R2+SKATERS_R3)
    k = random.choice([1,2,3,5,8,13,21])
    n_trials = random.choice([15,50])
    n_burn = PROPHET_META['n_warm']
    n = n_burn+100               # Length of synthetic data
    print('Skater is '+str(f.__name__))
    print('Using '+str(optimizer.__name__))
    r_star, best_val, info = optimal_r(f=f, y=brownian_with_exogenous(n=n),k=k,a=None,
                       n_trials=n_trials, optimizer=optimizer, n_burn=n_burn)
    print("Best hyper-param is " + str(r_star))
    print('Took ' + str( (time.time()-start_time)/60 ) + ' minutes.' )
    pprint(info)



if __name__=='__main__':
    optimize_random_skater()