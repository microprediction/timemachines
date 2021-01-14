from poap.controller import BasicWorkerThread, ThreadController
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import CubicKernel, LinearTail, RBFInterpolant
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


def pysot_cube( objective, n_trials, n_dim, with_count=False):
    """ Minimize
    :param objective:
    :param n_trials:
    :param n_dim:
    :param with_count:
    :return:
    """

    num_threads = 1
    max_evals = n_trials
    gp = GenericProblem(dim=n_dim, objective=objective)
    rbf = RBFInterpolant(dim=n_dim, lb=np.array([0.0]*n_dim), ub=np.array([1.0]*n_dim), kernel=CubicKernel(), tail=LinearTail(n_dim))
    slhd = SymmetricLatinHypercube(dim=n_dim, num_pts=2 * (n_dim + 1))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = SRBFStrategy(
        max_evals=max_evals, opt_prob=gp, exp_design=slhd, surrogate=rbf, asynchronous=True
    )

    # Launch the threads and give them access to the objective function
    for _ in range(num_threads):
        worker = BasicWorkerThread(controller, gp.eval)
        controller.launch_worker(worker)

    # Run the optimization strategy
    result = controller.run()
    best_x = result.params[0].tolist()
    return (result.value, best_x, gp.feval_count) if with_count else (result.value, best_x)


if __name__ == '__main__':
    from timemachines.optimizers.objectives import AN_OBJECTIVE
    print(pysot_cube(AN_OBJECTIVE, n_trials=100, n_dim=3, with_count=True))