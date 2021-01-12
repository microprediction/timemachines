from pprint import pprint
from timemachines.optimizers.compendium import OPTIMIZERS
from timemachines.optimizers.objectives import OBJECTIVES


def optimizer_name(solver):
    return solver.__name__.replace('_cube','').replace('_skew',' (skew) ').replace('_scaled',' (scaled) ')

def debug():
    for n_trials in [2,10,40]:
        for n_dim in [2,3]:
            for optimizer in OPTIMIZERS:
                for objective in OBJECTIVES:
                    try:
                        optimizer(objective, n_trials=n_trials,n_dim=n_dim,with_count=True)
                    except Exception as e:
                        print(e)
                        print(optimizer_name(optimizer) + ' fails on ' + objective.__name__ +
                              ' in dimension '+str(n_dim)+' when n_trials='+str(n_trials))


def comparison():
    n_trials,n_dim = 300, 5
    comparison = sorted([(objective.__name__,optimizer(objective, n_trials=n_trials, n_dim=n_dim, with_count=True),
                          optimizer_name(optimizer))
                         for optimizer in OPTIMIZERS for objective in OBJECTIVES])
    pprint(comparison)


if __name__=='__main__':
    # Beware that some optimizers may take more function evals than instructed
    # Others may multi-thread
    # This is here more as a quick unit test than a legitimate "comparison"
    comparison()





