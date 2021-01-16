from hyperopt import fmin, hp, tpe, Trials
from hyperopt.tpe import suggest as tpe_suggest
from hyperopt.rand import suggest as rand_suggest
from hyperopt.atpe import suggest as atpe_suggest


def hyperopt_cube(objective, n_trials, n_dim, with_count=False, algo=None):
    """ Minimize a function on the cube using HyperOpt, and audit # of function calls
       :param objective:    function on (0,1)^n_dim
       :param n_trials:     Guideline for function evaluations
       :param n_dim:
       :param with_count:
       :return:
    """
    assert algo is not None, 'provide algo'
    hp_space = dict([('u' + str(i), hp.uniform('u' + str(i), 0, 1)) for i in range(n_dim)])

    global feval_count
    feval_count = 0

    def _objective(hps):
        global feval_count
        feval_count += 1
        us = [hps['u' + str(i)] for i in range(n_dim)]
        return objective(us)

    trls = Trials()
    res = fmin(_objective, space=hp_space, algo=tpe.suggest, trials=trls, max_evals=n_trials, show_progressbar=False)
    best_x = [trls.best_trial['misc']['vals']['u' + str(i)][0] for i in range(n_dim)]
    best_val = trls.best_trial['result']['loss']
    return (best_val, best_x, feval_count) if with_count else (best_val, best_x)


def hyperopt_atpe_cube(objective, n_trials, n_dim, with_count=False):
    return hyperopt_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, algo=atpe_suggest)


def hyperopt_tpe_cube(objective, n_trials, n_dim, with_count=False):
    return hyperopt_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, algo=tpe_suggest)


def hyperopt_rand_cube(objective, n_trials, n_dim, with_count=False):
    return hyperopt_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, algo=rand_suggest)


HYPEROPT_OPTIMIZERS = [hyperopt_atpe_cube, hyperopt_rand_cube, hyperopt_tpe_cube]

if __name__ == '__main__':
    from timemachines.optimizers.objectives import OBJECTIVES

    for objective in OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in HYPEROPT_OPTIMIZERS:
            print(optimizer(objective, n_trials=50, n_dim=5, with_count=True))
