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
    return (trls.best_trial['result']['loss'], feval_count) if with_count else trls.best_trial['result']['loss']


