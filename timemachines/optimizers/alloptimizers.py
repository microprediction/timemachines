from timemachines.optimizers.hyperoptcube import HYPEROPT_OPTIMIZERS
from timemachines.optimizers.shgocube import SHGO_OPTIMIZERS
from timemachines.optimizers.optunacube import OPTUNA_OPTIMIZERS
from timemachines.optimizers.pysotcube import PYSOT_OPTIMIZERS
from timemachines.optimizers.scipycube import SCIPY_OPTIMIZERS
from timemachines.optimizers.axcube import AX_OPTIMIZERS
from timemachines.optimizers.platypuscube import PLATYPUS_OPTIMIZERS
from timemachines.optimizers.pymoocube import PYMOO_OPTMIZERS
from timemachines.optimizers.swarmlibcube import SWARMLIB_OPTIZERS
from timemachines.optimizers.nevergradcube import NEVERGRAD_OPTIMIZERS
from timemachines.skaters.components.chronometer import tick, tock

CANDIDATES = SCIPY_OPTIMIZERS + SHGO_OPTIMIZERS + HYPEROPT_OPTIMIZERS +\
             PYSOT_OPTIMIZERS + OPTUNA_OPTIMIZERS + AX_OPTIMIZERS +\
             PLATYPUS_OPTIMIZERS + PYMOO_OPTMIZERS + NEVERGRAD_OPTIMIZERS + SWARMLIB_OPTIZERS

# To see what might be working, or not, refer regression testing results in directories such as:
# https://github.com/microprediction/timemachines-testing/tree/main/data/brownian/dlm_seasonal
OPTIMIZERS = SHGO_OPTIMIZERS + SCIPY_OPTIMIZERS + PYSOT_OPTIMIZERS + AX_OPTIMIZERS + \
             OPTUNA_OPTIMIZERS + PLATYPUS_OPTIMIZERS + NEVERGRAD_OPTIMIZERS + SWARMLIB_OPTIZERS


def optimizer_from_name(name):
    valid = [f for f in OPTIMIZERS if f.__name__==name ]
    return valid[0] if len(valid)==1 else None



if __name__=='__main__':
    print(' ')
    print('Full list of optimizer strategies .. ')
    print(' ')
    print([ o.__name__.replace('_cube','') for o in OPTIMIZERS])
    print(' ')
    print(str(len(OPTIMIZERS)) + ' optimization strategies will be compared.')
    from timemachines.optimizers.objectives import OBJECTIVES
    print(str(len(OBJECTIVES)) + ' objective functions will be employed.')
    for objective in OBJECTIVES:
        print(' ')
        for n_dim in [2,6,20]:
            for n_trials in [20,150]:
                print(' Now testing against '+objective.__name__+' in '+str(n_dim)+' dimensions with '+str(n_trials)+' trials.')
                for optimizer in OPTIMIZERS:
                    try:
                        print(optimizer.__name__,(optimizer.__name__,optimizer(objective, n_trials=n_trials, n_dim=n_dim, with_count=True)))
                    except:
                        import warnings
                        print(' ')
                        warnings.warn(' WARNING : '+optimizer.__name__+' fails on '+ objective.__name__+ ' in '+str(n_dim)+' dimensions with '+str(n_trials)+' trials.')


