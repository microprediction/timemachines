from pymoo.optimize import minimize
from pymoo.model.problem import Problem
import numpy as np
from pymoo.factory import get_algorithm, get_termination, get_reference_directions

REF_DIRS = get_reference_directions("das-dennis", 1, n_partitions=6)


def pymoo_brkga_cube(objective, n_trials, n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="brkga", n_dim=n_dim, with_count=with_count)


def pymoo_nelder_cube(objective, n_trials, n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="nelder-mead", n_dim=n_dim, with_count=with_count)


def pymoo_cmaes_cube(objective, n_trials, n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="cmaes", n_dim=n_dim, with_count=with_count)


def pymoo_nsga2_cube(objective, n_trials, n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="nsga2", ref_dirs=REF_DIRS, n_dim=n_dim, with_count=with_count)


def pymoo_rnsga2_cube(objective, n_trials, n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="rnsga2", ref_dirs=REF_DIRS, n_dim=n_dim, with_count=with_count)


def pymoo_rnsga3_cube(objective, n_trials, n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="rnsga3", ref_dirs=REF_DIRS, n_dim=n_dim, with_count=with_count)


def pymoo_unsga3_cube(objective, n_trials, n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="unsga3", ref_dirs=REF_DIRS, n_dim=n_dim, with_count=with_count)


def pymoo_moead_cube(objective, n_trials, n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="moead", ref_dirs=REF_DIRS, n_dim=n_dim, with_count=with_count)


def pymoo_pattern_cube(objective, n_trials, n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="pattern-search", n_dim=n_dim, with_count=with_count)


def pymoo_ctaea_cube(objective, n_trials, n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="ctaea", ref_dirs=REF_DIRS, n_dim=n_dim, with_count=with_count)


def pymoo_nsga3_cube(objective, n_trials, n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="nsga3", ref_dirs=REF_DIRS, n_dim=n_dim, with_count=with_count)


def pymoo_de_cube(objective, n_trials, n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="nsga3", n_dim=n_dim, with_count=with_count)


def pymoo_cube(objective, n_trials, method_name, n_dim, with_count, ref_dirs=None):

    class ObjectiveProblem(Problem):

        def __init__(self):
            super().__init__(n_var=n_dim, n_obj=1, n_constr=0, xl=0.0, xu=1.0)
            self.feval_count = 0

        def _evaluate(self, x, out, *args, **kwargs):
            """ vectorized  """
            self.feval_count = self.feval_count + len(x)
            out["F"] = np.array([objective(u) for u in x])

    try:
        algorithm = get_algorithm(method_name, ref_dirs=ref_dirs)
    except ValueError:
        algorithm = get_algorithm(method_name)
    termination = get_termination("n_eval", n_trials)
    problem = ObjectiveProblem()

    result = minimize(problem=problem,
                      algorithm=algorithm,
                      termination=termination,
                      seed=None,
                      verbose=False,
                      display=None,
                      callback=None,
                      return_least_infeasible=False,
                      save_history=False
                      )
    best_val = result.F[0]
    best_x = result.X.tolist()
    return (best_val, best_x, problem.feval_count) if with_count else (best_val, best_x)

PYMOO_CANDIDATES = [ pymoo_de_cube, pymoo_nsga2_cube, pymoo_rnsga2_cube,
                     pymoo_nelder_cube, pymoo_ctaea_cube,
                     pymoo_nsga3_cube, pymoo_rnsga3_cube, pymoo_unsga3_cube,
                     pymoo_pattern_cube, pymoo_brkga_cube, pymoo_nsga2_cube ]

PYMOO_OPTMIZERS = [ pymoo_nsga2_cube, pymoo_nelder_cube, pymoo_nsga3_cube,
                    pymoo_unsga3_cube,pymoo_pattern_cube, pymoo_brkga_cube,
                    pymoo_nsga2_cube]

# TODO: See why ['pymoo_ctaea_cube', 'pymoo_de_cube', 'pymoo_rnsga2_cube', 'pymoo_rnsga3_cube'] are broken sometimes

if __name__ == '__main__':
    from timemachines.optimizers.objectives import OBJECTIVES
    always_working = PYMOO_CANDIDATES
    broken = set()
    for objective in OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in PYMOO_CANDIDATES:
            try:
                print(optimizer(objective, n_trials=50, n_dim=5, with_count=True))
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

