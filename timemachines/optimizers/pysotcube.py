from poap.controller import BasicWorkerThread, ThreadController
from pySOT.experimental_design import SymmetricLatinHypercube, LatinHypercube, TwoFactorial
from pySOT.strategy import SRBFStrategy, EIStrategy, DYCORSStrategy,RandomStrategy, LCBStrategy
from pySOT.surrogate import CubicKernel, LinearTail, RBFInterpolant, GPRegressor
from pySOT.optimization_problems.optimization_problem import OptimizationProblem
import numpy as np


class GenericProblem(OptimizationProblem):

    def __init__(self, dim, objective):
        self.dim = dim
        self.min = 0
        self.minimum = np.zeros(dim)
        self.lb = 0 * np.ones(dim)
        self.ub = 1 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.objective = objective
        self.info = str(dim) + "-dimensional objective function " + objective.__name__
        self.feval_count = 0

    def eval(self, x):
        """Evaluate the objective x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        d = float(self.dim)
        self.feval_count = self.feval_count+1
        return self.objective(list(x))


def pysot_cube( objective, n_trials, n_dim, with_count=False, method=None, design=None):
    """ Minimize
    :param objective:
    :param n_trials:
    :param n_dim:
    :param with_count:
    :return:
    """

    num_threads = 1
    asynchronous = True

    max_evals = n_trials
    gp = GenericProblem(dim=n_dim, objective=objective)

    if design=='latin':
        exp_design = LatinHypercube(dim=n_dim, num_pts=2 * (n_dim + 1))
    elif design=='symmetric':
        exp_design = SymmetricLatinHypercube(dim=n_dim, num_pts=2 * (n_dim + 1))
    elif design=='factorial':
        exp_design = TwoFactorial(dim=n_dim)
    else:
        raise ValueError('design should be latin, symmetric or factorial')

    # Create a strategy and a controller
    #  SRBFStrategy, EIStrategy, DYCORSStrategy,RandomStrategy, LCBStrategy
    controller = ThreadController()
    if method.lower()=='srbf':
        surrogate = RBFInterpolant(dim=n_dim, lb=np.array([0.0] * n_dim), ub=np.array([1.0] * n_dim), kernel=CubicKernel(),
                             tail=LinearTail(n_dim))
        controller.strategy = SRBFStrategy(
            max_evals=max_evals, opt_prob=gp, exp_design=exp_design, surrogate=surrogate, asynchronous=asynchronous
        )
    elif method.lower() == 'ei':
        surrogate = GPRegressor(dim=n_dim, lb=np.array([0.0] * n_dim), ub=np.array([1.0] * n_dim) )
        controller.strategy = EIStrategy(
            max_evals=max_evals, opt_prob=gp, exp_design=exp_design, surrogate=surrogate, asynchronous=asynchronous
        )
    elif method.lower() == 'dycors':
        surrogate = RBFInterpolant(dim=n_dim, lb=np.array([0.0] * n_dim), ub=np.array([1.0] * n_dim), kernel=CubicKernel(),
                             tail=LinearTail(n_dim))
        controller.strategy = DYCORSStrategy(
            max_evals=max_evals, opt_prob=gp, exp_design=exp_design, surrogate=surrogate, asynchronous=asynchronous
        )
    elif method.lower() == 'lcb':
        surrogate = GPRegressor(dim=n_dim, lb=np.array([0.0] * n_dim), ub=np.array([1.0] * n_dim))
        controller.strategy = LCBStrategy(
            max_evals=max_evals, opt_prob=gp, exp_design=exp_design, surrogate=surrogate, asynchronous=asynchronous
        )
    elif method.lower() == 'random':
        controller.strategy = RandomStrategy(
            max_evals=max_evals, opt_prob=gp
        )
    else:
        raise ValueError("Didn't recognize method passed to pysot")


    # Launch the threads and give them access to the objective function
    for _ in range(num_threads):
        worker = BasicWorkerThread(controller, gp.eval)
        controller.launch_worker(worker)

    # Run the optimization strategy
    result = controller.run()
    best_x = result.params[0].tolist()
    return (result.value, best_x, gp.feval_count) if with_count else (result.value, best_x)


# Index by stratgegy
# SRBFStrategy, EIStrategy, DYCORSStrategy,RandomStrategy, LCBStrategy

def pysot_srbf_cube( objective, n_trials, n_dim, with_count=False):
    return pysot_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='srbf', design='symmetric' )


def pysot_ei_cube( objective, n_trials, n_dim, with_count=False):
    return pysot_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='ei', design='symmetric' )


def pysot_dycors_cube( objective, n_trials, n_dim, with_count=False):
    return pysot_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='dycors', design='symmetric' )


def pysot_lcb_cube( objective, n_trials, n_dim, with_count=False):
    return pysot_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='lcb', design='symmetric' )


def pysot_random_cube( objective, n_trials, n_dim, with_count=False):
    return pysot_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='random', design='symmetric' )


PYSOT_OPTIMIZERS = [ pysot_ei_cube, pysot_lcb_cube, pysot_random_cube, pysot_srbf_cube, pysot_dycors_cube ]


if __name__ == '__main__':
    from timemachines.optimizers.objectives import OBJECTIVES
    for objective in OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in PYSOT_OPTIMIZERS:
            print( (optimizer.__name__, optimizer(objective, n_trials=100, n_dim=3, with_count=True)))