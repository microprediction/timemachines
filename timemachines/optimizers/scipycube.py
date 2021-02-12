from scipy.optimize import minimize
from timemachines.objectives.classic import CLASSIC_OBJECTIVES
global feval_count

# TODO: standardize with SHGO and remove duplicate code
from copy import deepcopy
MINIMIZER_KWARGS = {'slqsp': {'method': 'SLQSP'},
                    'powell': {'method': 'Powell'},
                    'nelder': {'method': 'Nelder-Mead'},
                    'dogleg': {'method': 'dogleg'}
                    }


def scipy_cube(objective, n_trials, n_dim, with_count=False, method=None):
    bounds = [(0,1) ]*n_dim

    options = deepcopy(MINIMIZER_KWARGS[method])
    options.update({'maxfev': n_trials})

    global feval_count
    feval_count = 0

    def _objective(x):
        global feval_count
        feval_count +=1
        return objective(list(x))

    result = minimize(_objective, x0=[0]*n_dim, method='powell',bounds=bounds, options=options)
    best_x = result.x.tolist()
    best_val = _objective(result.x)
    return (best_val, best_x,  feval_count) if with_count else (best_val, best_x)


def scipy_slqsp_cube(objective, n_trials, n_dim, with_count=False ):
    return scipy_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='slqsp')


def scipy_powell_cube(objective, n_trials, n_dim, with_count=False):
    return scipy_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='powell')


def scipy_nelder_cube(objective, n_trials, n_dim, with_count=False):
    return scipy_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='nelder')


def scipy_dogleg_cube(objective, n_trials, n_dim, with_count=False):
    return scipy_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='dogleg')


SCIPY_OPTIMIZERS = [ scipy_slqsp_cube, scipy_powell_cube, scipy_nelder_cube, scipy_dogleg_cube ]


if __name__ == '__main__':
    for objective in CLASSIC_OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in SCIPY_OPTIMIZERS:
            print((optimizer.__name__,(optimizer(objective, n_trials=250, n_dim=6, with_count=True))))

