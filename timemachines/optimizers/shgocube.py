from scipy.optimize import shgo

# Define how SHGO does local search
MINIMIZER_KWARGS = {'slqsp': {'method': 'SLQSP',
                              'max_iter': 10},
                    'powell': {'method': 'Powell',
                               'max_iter': 10},
                    'nelder': {'method': 'Nelder-Mead',
                               'maxiter': 10},
                    'dogleg': {'method': 'dogleg',
                               'maxiter': 10}
                    }


def shgo_slqsp_sobol_cube(objective, n_trials, n_dim, with_count: bool = False):
    return shgo_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                     local_method='slqsp', sampling_method='sobol')


def shgo_slqsp_simplicial_cube(objective, n_trials, n_dim, with_count: bool = False):
    return shgo_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                     local_method='slqsp', sampling_method='simplicial')


def shgo_powell_sobol_cube(objective, n_trials, n_dim, with_count: bool = False):
    return shgo_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                     local_method='powell', sampling_method='sobol')


def shgo_powell_simplicial_cube(objective, n_trials, n_dim, with_count: bool = False):
    return shgo_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                     local_method='powell', sampling_method='simplicial')


def shgo_nelder_sobol_cube(objective, n_trials, n_dim, with_count: bool = False):
    return shgo_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                     local_method='nelder', sampling_method='sobol')


def shgo_nelder_simplicial_cube(objective, n_trials, n_dim, with_count: bool = False):
    return shgo_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                     local_method='nelder', sampling_method='simplicial')


def shgo_dogleg_sobol_cube(objective, n_trials, n_dim, with_count: bool = False):
    return shgo_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                     local_method='dogleg', sampling_method='sobol')


def shgo_dogleg_simplicial_cube(objective, n_trials, n_dim, with_count: bool = False):
    return shgo_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                     local_method='dogleg', sampling_method='simplicial')


SHGO_OPTIMIZERS = [shgo_slqsp_sobol_cube, shgo_slqsp_simplicial_cube, shgo_powell_sobol_cube,
                   shgo_powell_simplicial_cube, shgo_nelder_sobol_cube, shgo_nelder_simplicial_cube,
                   shgo_dogleg_sobol_cube, shgo_dogleg_simplicial_cube]


def shgo_cube(objective, n_trials, n_dim, with_count: bool = False, local_method=None, sampling_method='sobol'):
    """ Minimize a function on the cube using SHGO
    :param objective:    function on (0,1)^n_dim
    :param n_trials:
    :param n_dim:
    :param with_count:
    :return:
    """
    minimizer_kwargs = MINIMIZER_KWARGS[local_method]
    assert sampling_method in ['sobol', 'simplicial'], ' did not understand sampling method'
    bounds = [(0, 1)] * n_dim

    global feval_count
    feval_count = 0

    def _objective(x):
        global feval_count
        feval_count += 1
        return objective(list(x))

    # Try to induce roughly the right number of function evaluations. This can be improved!
    n_trials_reduced = n_trials - 20
    n_iters = int(1 + n_trials / 40)
    n = int(5 + n_trials / 20)
    result = shgo(_objective, bounds, n=n, iters=n_iters, options={'maxfev': n_trials_reduced,
                                                                   'minimize_every_iter': False,
                                                                   'maxfun': n_trials_reduced,
                                                                   'minimizer_kwargs': minimizer_kwargs},
                  sampling_method=sampling_method)
    return (result.fun, list(result.x), feval_count) if with_count else (result.fun, result.x)


if __name__ == '__main__':
    from timemachines.optimizers.objectives import OBJECTIVES
    for objective in OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in SHGO_OPTIMIZERS:
            print(optimizer(objective, n_trials=50, n_dim=5, with_count=True))
