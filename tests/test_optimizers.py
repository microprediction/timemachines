from timemachines.optimizers.alloptimizers import OPTIMIZERS
from timemachines.optimizers.objectives import OBJECTIVES


def test_compendium():
    n_trials = 10
    for n_dim in [2,3]:
        for optimizer in OPTIMIZERS:
            for objective in OBJECTIVES:
                try:
                    optimizer(objective, n_trials=n_trials,n_dim=n_dim,with_count=True)
                except Exception as e:
                    print(e)
                    raise Exception(optimizer.__name__ + ' fails on ' + objective.__name__)
