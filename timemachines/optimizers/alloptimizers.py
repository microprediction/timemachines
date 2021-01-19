from timemachines.optimizers.hyperoptcube import HYPEROPT_OPTIMIZERS
from timemachines.optimizers.shgocube import SHGO_OPTIMIZERS
from timemachines.optimizers.optunacube import OPTUNA_OPTIMIZERS
from timemachines.optimizers.pysotcube import PYSOT_OPTIMIZERS
from timemachines.optimizers.scipycube import SCIPY_OPTIMIZERS
from timemachines.optimizers.axcube import AX_OPTIMIZERS
from timemachines.optimizers.platypuscube import PLATYPUS_OPTIMIZERS
from timemachines.optimizers.pymoocube import PYMOO_OPTMIZERS

CANDIDATES = SCIPY_OPTIMIZERS + SHGO_OPTIMIZERS + HYPEROPT_OPTIMIZERS +\
             PYSOT_OPTIMIZERS + OPTUNA_OPTIMIZERS + AX_OPTIMIZERS +\
             PLATYPUS_OPTIMIZERS + PYMOO_OPTMIZERS

# To see what might be working, or not, refer to directories such as:
# https://github.com/microprediction/timemachines-testing/tree/main/data/brownian/dlm_seasonal
OPTIMIZERS = SCIPY_OPTIMIZERS + PYSOT_OPTIMIZERS + AX_OPTIMIZERS + OPTUNA_OPTIMIZERS



if __name__=='__main__':
    print(str(len(OPTIMIZERS)) + ' in use')
    from timemachines.optimizers.objectives import OBJECTIVES
    for objective in OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in OPTIMIZERS:
            print(optimizer.__name__,(optimizer.__name__,optimizer(objective, n_trials=50, n_dim=5, with_count=True)))
