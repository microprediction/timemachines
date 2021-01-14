from timemachines.optimization import optimize
from timemachines.synthetic import brownian_with_exogenous
from timemachines.conventions import from_space
from timemachines.optimizers.alloptimizers import OPTIMIZERS
from timemachines.skaters.pmd import pmd_hyperparams, pmd_auto
from pprint import pprint
import random


def optimize_dlm():
    optimizer = random.choice(OPTIMIZERS)
    print('Running '+str(optimizer.__name__))
    best_val, best_x = optimize(f=pmd_auto, ys=brownian_with_exogenous(n=300),
                      n_trials=25, n_dim=3, optimizer=optimizer)
    best_r = from_space(best_x)
    s = pmd_hyperparams(s=dict(), r=best_r)
    print("Best hyper-param is " + str(best_r)+' with interpretation ')
    pprint(s)