from timemachines.optimization import optimize
from timemachines.synthetic import brownian_with_exogenous
from timemachines.optimizers.compendium import OPTIMIZERS
from timemachines.skaters.pmd import pmd_hyperparams, pmd_auto
from pprint import pprint
import random


def optimize_dlm():
    optimizer = random.choice(OPTIMIZERS)
    print('Running '+str(optimizer.__name__))
    r_star = optimize(f=pmd_auto, ys=brownian_with_exogenous(n=300),
                      n_trials=25, optimizer=optimizer)
    s = pmd_hyperparams(s=dict(), r=r_star)
    print("Best hyper-param is " + str(r_star)+' with interpretation ')
    pprint(s)