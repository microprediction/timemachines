from scipy.optimize import shgo


def shgo_cube(objective, n_trials, n_dim, with_count: bool = False):
    """ Minimize a function on the cube using SHGO
    :param objective:    function on (0,1)^n_dim
    :param n_trials:
    :param n_dim:
    :param with_count:
    :return:
    """
    bounds = [(0, 1)] * n_dim

    global feval_count
    feval_count = 0

    def _objective(x):
        global feval_count
        feval_count += 1
        return objective(list(x))

    # Try to induce roughly the right number of function evaluations
    n_trials_reduced = n_trials - 20
    n_iters = int(1+n_trials/40)
    n = int(5+n_trials/20)
    result = shgo(_objective, bounds, n=n, iters=n_iters, options={'maxfev': n_trials_reduced,
                                               'minimize_every_iter': False,
                                               'maxfun': n_trials_reduced,
                                               'minimizer_kwargs':{'maxiter':10}},
                                sampling_method='sobol')
    return (result.fun, feval_count) if with_count else result.fun


if __name__ == '__main__':
    from timemachines.optimizers.objectives import OBJECTIVES

    for objective in OBJECTIVES:
        print(shgo_cube(objective, n_trials=50, n_dim=5, with_count=True))
