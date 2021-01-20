from platypus import NSGAII, Problem, Real, EvolutionaryStrategy, GeneticAlgorithm,\
    NSGAIII, CMAES, GDE3, IBEA, MOEAD, OMOPSO, SMPSO, SPEA2, EpsMOEA, normal_boundary_weights
from platypus.core import FixedLengthArray

# Platypus is designed for multi-objective optimization, so may not do so well on single-objective problems

PLATYPUS_ALGORITHMS = {'evolutionary':EvolutionaryStrategy,
                       'genetic':GeneticAlgorithm,
                       'nsgaii':NSGAII,
                      'nsgaiii':(NSGAIII, {"divisions_outer": 12}),
                      'cmaes':(CMAES, {"epsilons": [0.05]}),
                      'gde3':GDE3,
                      'ibea':IBEA,
                      'moead':(MOEAD, {"weight_generator": normal_boundary_weights, "divisions_outer": 12}),
                      'omopso':(OMOPSO, {"epsilons": [0.05]}),
                      'smpso':SMPSO,
                      'spea2':SPEA2,
                      'epsmoea':(EpsMOEA, {"epsilons": [0.05]})
                      }


def platypus_cube(objective, n_trials, n_dim, with_count=False, method=None):
    global feval_count
    feval_count = 0

    def _objective(vars):
        global feval_count
        feval_count += 1
        return float(objective(list(vars))) # Avoid np.array as Platypus may puke

    problem = Problem(n_dim, 1, 0)
    problem.types[:] = [Real(0.0, 1.0)] * n_dim
    problem.constraints[:] = "<=0"
    problem.function = _objective

    strategy_and_args = PLATYPUS_ALGORITHMS[method]
    if isinstance(strategy_and_args,tuple):
        strategy = strategy_and_args[0]
        strategy_args = strategy_and_args[1]
        algorithm = strategy(problem, **strategy_args)
    else:
        strategy = strategy_and_args
        algorithm = strategy(problem)

    algorithm.run(n_trials)
    feasible_solution_obj = sorted([(s.objectives[0], s.variables) for s in algorithm.result if s.feasible],
                                   reverse=False)
    best_obj, best_x = feasible_solution_obj[0]
    if isinstance(best_x,FixedLengthArray):
        best_x = best_x._data   # CMA-ES returns it this way for some reason
    return (best_obj, best_x, feval_count) if with_count else (best_obj, best_x)


def platypus_evolutionary_cube(objective, n_trials, n_dim, with_count=False):
    return platypus_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                         method='genetic')


def platypus_genetic_cube(objective, n_trials, n_dim, with_count=False):
    return platypus_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                         method='genetic')

def platypus_nsgaii_cube(objective, n_trials, n_dim, with_count=False):
    return platypus_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                         method='nsgaii')

def platypus_cmaes_cube(objective, n_trials, n_dim, with_count=False):
    return platypus_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                         method='cmaes')

def platypus_gde3_cube(objective, n_trials, n_dim, with_count=False):
    return platypus_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                         method='gde3')

def platypus_ibea_cube(objective, n_trials, n_dim, with_count=False):
    return platypus_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                         method='ibea')

def platypus_moead_cube(objective, n_trials, n_dim, with_count=False):
    return platypus_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                         method='moead')

def platypus_omopso_cube(objective, n_trials, n_dim, with_count=False):
    return platypus_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                         method='omopso')

def platypus_smpso_cube(objective, n_trials, n_dim, with_count=False):
    return platypus_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                         method='smpso')

def platypus_spea2_cube(objective, n_trials, n_dim, with_count=False):
    return platypus_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                         method='spea2')

def platypus_epsmoea_cube(objective, n_trials, n_dim, with_count=False):
    return platypus_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count,
                         method='epsmoea')


PLATYPUS_OPTIMIZERS = [platypus_genetic_cube, platypus_evolutionary_cube, platypus_nsgaii_cube,
                       platypus_nsgaii_cube, platypus_cmaes_cube, platypus_gde3_cube,
                       platypus_ibea_cube, platypus_moead_cube, platypus_omopso_cube,
                       platypus_smpso_cube, platypus_spea2_cube, platypus_epsmoea_cube]


if __name__ == '__main__':
    from timemachines.optimizers.objectives import OBJECTIVES

    for objective in OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in PLATYPUS_OPTIMIZERS:
            print((optimizer.__name__, optimizer(objective, n_trials=50, n_dim=5, with_count=True)))
