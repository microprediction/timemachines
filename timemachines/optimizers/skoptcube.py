from timemachines.objectives.classic import CLASSIC_OBJECTIVES
from skopt import gp_minimize
global feval_count
from copy import deepcopy
import traceback

GP_INTERPRETATIONS = dict([(ipg, {'initial_point_generator':ipg}) for ipg in ["random", "sobol", "hammersly", "lhs"]])
GP_INTERPRETATIONS.update({  'lcbexplore':{"acq_func": "LCB", "kappa":3.0},
                             'lcb':{"acq_func":"LCB","kappa":1.96},
                             'lcbexploit':{"acq_func":"LCB","kappa":1.0},
                             'pi':{"acq_func":"pi"},
                             'xi02':{"xi":0.02},
                             'default':{},
                             'sampling':{'acq_optimizer':'sampling'},
                             'lbfgs':{'acq_optimizer':'lbfgs'},
                             'noisy':{'noise':'gaussian'},
                             'calm':{'noise':1e-8},
                             '':{}
                           })


def skopt_gp_method_to_kwargs(method:str)->dict:
    """ Create param list for skopt.gp_minimize from name of strategy
         e.g.  lcbexplore_sampling_sobol -> {'acq_func':'LCB',... etc}
    """
    cues = method.split('_')
    params = dict()
    for cue in cues:
        if cue:
            if GP_INTERPRETATIONS.get(cue):
                params.update(deepcopy(GP_INTERPRETATIONS[cue]))
            else:
                msg = 'skopt_gp cannot interpret the cue ' + cue
                raise ValueError(msg)
    return params


def skopt_gp_cube_factory(objective, n_trials, n_dim, with_count=False, method=''):
    bounds = [(0,1) ]*n_dim

    global feval_count
    feval_count = 0

    def _objective(x):
        global feval_count
        feval_count +=1
        return objective(list(x))

    gp_params = skopt_gp_method_to_kwargs(method=method)
    result = gp_minimize(func=_objective,  dimensions=bounds, n_calls=n_trials, n_jobs=1, **gp_params)
    best_x = list(result.x)
    best_val = result.fun
    return (best_val, best_x,  feval_count) if with_count else (best_val, best_x)


# We could now create the following functions programmatically, but it is somewhat easier to debug this way

def skopt_gp_default_cube(objective, n_trials, n_dim, with_count=False):
    return skopt_gp_cube_factory(objective=objective,n_trials=n_trials,n_dim=n_dim, with_count=with_count,
                                 method='')


def skopt_gp_sobol_cube(objective, n_trials, n_dim, with_count=False):
        return skopt_gp_cube_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                                     method='sobol')


def skopt_gp_hammersley_cube(objective, n_trials, n_dim, with_count=False):
    return skopt_gp_cube_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                                 method='hammersly')


def skopt_gp_lcb_cube(objective, n_trials, n_dim, with_count=False):
    return skopt_gp_cube_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                                 method='lcb')


def skopt_gp_lcbexplore_cube(objective, n_trials, n_dim, with_count=False):
    return skopt_gp_cube_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                                 method='lcbexplore')


def skopt_gp_lcbexploit_cube(objective, n_trials, n_dim, with_count=False):
    return skopt_gp_cube_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                                 method='lcbexploit')


def skopt_gp_sampling_cube(objective, n_trials, n_dim, with_count=False):
    return skopt_gp_cube_factory(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                                 method='sampling')


SKOPT_GP_CANDIDATES = [ skopt_gp_default_cube, skopt_gp_sobol_cube, skopt_gp_hammersley_cube,
                        skopt_gp_lcb_cube, skopt_gp_lcbexplore_cube, skopt_gp_lcbexploit_cube,
                        skopt_gp_sampling_cube ]

SKOPT_GP_OPTIMIZERS = [] # Not ready for prime time?

if __name__ == '__main__':
    always_working = SKOPT_GP_CANDIDATES
    broken = set()
    for objective in CLASSIC_OBJECTIVES[:1]:
        print(' ')
        print(objective.__name__)
        for optimizer in SKOPT_GP_CANDIDATES:
            try:
                print(optimizer(objective, n_trials=250, n_dim=6, with_count=True))
            except Exception as e:
                print(e)
                broken.add(optimizer)
                always_working.remove(optimizer)
    print(' ')
    print('Sometimes broken: ')
    print([b.__name__ for b in broken])
    print(' ')
    print('Alway working: ')
    print([b.__name__ for b in always_working])



