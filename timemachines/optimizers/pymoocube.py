from pymoo.optimize import minimize
from pymoo.model.problem import Problem
import numpy as np
from pymoo.factory import get_algorithm, get_termination, get_reference_directions

REF_DIRS = get_reference_directions("das-dennis", 1, n_partitions=6)


def brkga_cube(objective, n_trials,n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="brkga", n_dim=n_dim, with_count=with_count)


def nelder_cube(objective, n_trials,n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="nelder-mead", n_dim=n_dim, with_count=with_count)


def cmaes_cube(objective, n_trials,n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="cmaes", n_dim=n_dim, with_count=with_count)


def nsga2_cube(objective, n_trials,n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="nsga2", ref_dirs=REF_DIRS, n_dim=n_dim, with_count=with_count)


def rnsga2_cube(objective, n_trials,n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="rnsga2", ref_dirs=REF_DIRS, n_dim=n_dim, with_count=with_count)


def rnsga3_cube(objective, n_trials,n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="rnsga3", ref_dirs=REF_DIRS, n_dim=n_dim, with_count=with_count)


def unsga3_cube(objective, n_trials,n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="unsga3", ref_dirs=REF_DIRS, n_dim=n_dim, with_count=with_count)


def moead_cube(objective, n_trials,n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="moead", ref_dirs=REF_DIRS, n_dim=n_dim, with_count=with_count)


def pattern_cube(objective, n_trials,n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="pattern-search", n_dim=n_dim, with_count=with_count)


def ctaea_cube(objective, n_trials,n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="ctaea", ref_dirs=REF_DIRS, n_dim=n_dim, with_count=with_count)


def nsga3_cube(objective,  n_trials, n_dim, with_count=False):
    return pymoo_cube(objective=objective,    n_trials=n_trials, method_name="nsga3", ref_dirs=REF_DIRS, n_dim=n_dim, with_count=with_count)


def de_cube(objective, n_trials, n_dim, with_count=False):
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
    f_min = result.F[0]
    best_x = result.X.tolist()
    return (f_min, best_x, problem.feval_count) if with_count else (f_min, best_x)


PYMOO_BROKEN = [de_cube,rnsga2_cube]
PYMOO_GOOD = [nelder_cube,ctaea_cube,nsga3_cube,pattern_cube]
PYMOO_BAD = [brkga_cube,nsga2_cube]

if __name__ == '__main__':
    from timemachines.optimizers.objectives import OBJECTIVES
    for solver in PYMOO_BAD:
        print(' ')
        for objective in OBJECTIVES:
            print((objective.__name__,solver(objective, n_trials=100, n_dim=3, with_count=True)))
    print('done')