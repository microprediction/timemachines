from timemachines.optimizers.objectives import OBJECTIVES
import nevergrad as ng

# Facebook nevergrad https://facebookresearch.github.io/nevergrad/optimization.html


NEVER_ENOUGH = {'ngopt':'nGOpt is “meta”-optimizer which adapts to the provided settings (budget, number of workers, parametrization) and should therefore be a good default.',
                'de':'TwoPointsDE is excellent in many cases, including very high num_workers.',
                'portfolio':'PortfolioDiscreteOnePlusOne is excellent in discrete settings of mixed settings when high precision on parameters is not relevant. Its possibly a good choice for hyperparameter choice.',
                'oneplus':"OnePlusOne is a simple robust method for continuous parameters with num_workers less than 8",
                'cma':'CMA is excellent for control (e.g. neurocontrol) when the environment is not very noisy (num_workers ~50 ok) and when the budget is large (e.g. 1000 x the dimension).',
                'tbpsa':'TBPSA is excellent for problems corrupted by noise, in particular overparameterized (neural) ones; very high num_workers ok).',
                'pso':'PSO is excellent in terms of robustness, high num_workers ok.',
                'hammersley':'ScrHammersleySearchPlusMiddlePoint is excellent for super parallel cases (fully one-shot, i.e. num_workers = budget included) or for very multimodal cases (such as some of our MLDA problems); don’t use softmax with this optimizer.',
                'random':'RandomSearch is the classical random search baseline; don’t use softmax with this optimizer.'}


def nevergrad_cube(objective, n_trials,n_dim, with_count=False, method=None):

    instrument = ng.p.Array(shape=(n_dim,)).set_bounds(lower=0.0,upper=1.0)
    num_workers = 1

    if method.lower()=='ngopt':
        optimizer = ng.optimizers.NGOpt(parametrization=instrument, budget=n_trials, num_workers=num_workers)
    elif method.lower()=='ngopt4':
        optimizer = ng.optimizers.NGOpt4(parametrization=instrument, budget=n_trials,  num_workers=num_workers)
    elif method.lower() == 'ngopt8':
        optimizer = ng.optimizers.NGOpt8(parametrization=instrument, budget=n_trials,  num_workers=num_workers)
    elif method.lower() == 'de':
        optimizer = ng.optimizers.TwoPointsDE(parametrization=instrument, budget=n_trials, num_workers=num_workers)
    elif method.lower() == 'portfolio':
        optimizer = ng.optimizers.Portfolio(parametrization=instrument, budget=n_trials,  num_workers=num_workers)
    elif method.lower() == 'oneplus':
        optimizer = ng.optimizers.OnePlusOne(parametrization=instrument, budget=n_trials,  num_workers=num_workers)
    elif method.lower() == 'tbpsa':
        optimizer = ng.optimizers.TBPSA(parametrization=instrument, budget=n_trials, num_workers=num_workers)
    elif method.lower() == 'hammersley':
        optimizer = ng.optimizers.ScrHammersleySearchPlusMiddlePoint(parametrization=instrument, budget=n_trials,
                                        num_workers=num_workers)
    elif method.lower() == 'pso':
        optimizer = ng.optimizers.PSO(parametrization=instrument, budget=n_trials, num_workers=num_workers)
    elif method.lower() == 'cma':
        optimizer = ng.optimizers.CMA(parametrization=instrument, budget=n_trials, num_workers=num_workers)
    elif method.lower() == 'random':
        optimizer = ng.optimizers.RandomSearch(parametrization=instrument, budget=n_trials, num_workers=num_workers)
    else:
        raise Exception('Not recognizing '+str(method))
    global feval_count
    feval_count = 0

    def cube_objective(us):
        global feval_count
        feval_count += 1
        return objective(us)

    recommendation = optimizer.minimize( cube_objective )
    best_x = recommendation.value
    best_val = recommendation.loss

    if with_count:
        return best_val, best_x, feval_count
    else:
        return best_val, best_x


def nevergrad_ngopt_cube(objective, n_trials,n_dim, with_count=False):
    return nevergrad_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='ngopt')


def nevergrad_ngopt4_cube(objective, n_trials,n_dim, with_count=False):
    return nevergrad_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='ngopt4')


def nevergrad_ngopt8_cube(objective, n_trials,n_dim, with_count=False):
    return nevergrad_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='ngopt8')


def nevergrad_de_cube(objective, n_trials,n_dim, with_count=False):
    return nevergrad_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='de')


def nevergrad_portfolio_cube(objective, n_trials,n_dim, with_count=False):
    return nevergrad_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='portfolio')


def nevergrad_oneplus_cube(objective, n_trials,n_dim, with_count=False):
    return nevergrad_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='oneplus')


def nevergrad_cma_cube(objective, n_trials,n_dim, with_count=False):
    return nevergrad_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='cma')


def nevergrad_hammersley_cube(objective, n_trials,n_dim, with_count=False):
    return nevergrad_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='hammersley')


NEVERGRAD_OPTIMIZERS = [ nevergrad_ngopt_cube, nevergrad_ngopt4_cube, nevergrad_ngopt8_cube,
                         nevergrad_de_cube, nevergrad_portfolio_cube, nevergrad_oneplus_cube,
                         nevergrad_cma_cube, nevergrad_hammersley_cube ]


if __name__=='__main__':
    for objective in OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in NEVERGRAD_OPTIMIZERS:
            print((optimizer.__name__,optimizer(objective, n_trials=100, n_dim=6, with_count=True)))
