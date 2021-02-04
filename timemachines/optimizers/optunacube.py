import optuna
from optuna.logging import CRITICAL
from timemachines.optimizers.objectives import OBJECTIVES


def optuna_cube_factory(objective, n_trials, n_dim, with_count=False, method=None):

    if method.lower()=='random':
        sampler = optuna.samplers.RandomSampler()
    elif method.lower()=='cmaes':
        sampler = optuna.samplers.CmaEsSampler()
    elif method.lower()=='tpe':
        sampler = optuna.samplers.TPESampler()
    else:
        raise ValueError('random, cmaes, tpe or grid please')

    global feval_count
    feval_count = 0

    def cube_objective(trial):
        global feval_count
        us = [ trial.suggest_float('u'+str(i),0,1) for i in range(n_dim)]
        feval_count += 1
        return objective(us)

    optuna.logging.set_verbosity(CRITICAL)
    study = optuna.create_study(sampler=sampler)
    study.optimize(cube_objective,n_trials=n_trials)

    best_x = [ study.best_params['u'+str(i)] for i in range(n_dim) ]

    if with_count:
        return study.best_value, best_x, feval_count
    else:
        return study.best_value, best_x


def optuna_random_cube(objective, n_trials,n_dim, with_count=False):
    return optuna_cube_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='random')


def optuna_cmaes_cube(objective, n_trials, n_dim, with_count=False):
    return optuna_cube_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='cmaes')


def optuna_tpe_cube(objective, n_trials, n_dim, with_count=False):
    return optuna_cube_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='tpe')


OPTUNA_OPTIMIZERS = [ optuna_cmaes_cube, optuna_tpe_cube, optuna_random_cube ]


if __name__=='__main__':
    for objective in OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in OPTUNA_OPTIMIZERS:
            print((optimizer(objective, n_trials=100, n_dim=6, with_count=True)))
