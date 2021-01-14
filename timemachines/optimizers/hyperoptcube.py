from hyperopt import fmin, hp, tpe, Trials


def hyperopt_cube(objective, n_trials, n_dim, with_count=False):
    """ Minimize a function on the cube using HyperOpt, and audit # of function calls
       :param objective:    function on (0,1)^n_dim
       :param n_trials:     Guideline for function evaluations
       :param n_dim:
       :param with_count:
       :return:
    """
    hp_space = dict( [('u'+str(i), hp.uniform('u'+str(i), 0, 1)) for i in range(n_dim)])

    global feval_count
    feval_count = 0

    def _objective(hps):
        global feval_count
        feval_count += 1
        us = [ hps['u'+str(i)] for i in range(n_dim)]
        return objective(us)

    trls = Trials()
    res = fmin(_objective, space=hp_space, algo=tpe.suggest, trials=trls, max_evals=n_trials, show_progressbar=False)
    best_x = [ trls.best_trial['misc']['vals']['u'+str(i)][0] for i in range(n_dim) ]
    best_val = trls.best_trial['result']['loss']
    return (best_val, best_x, feval_count) if with_count else (best_val,best_x)



if __name__ == '__main__':
    from timemachines.optimizers.objectives import OBJECTIVES
    for objective in OBJECTIVES:
        print(hyperopt_cube(objective, n_trials=50, n_dim=5, with_count=True))