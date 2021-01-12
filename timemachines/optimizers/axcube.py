from ax import optimize
from logging import CRITICAL
from ax.utils.common.logger import get_logger
rt = get_logger('ax')
rt.setLevel(CRITICAL)
import warnings
warnings.filterwarnings("ignore")
from funcy import print_durations
from timemachines.optimizers.objectives import OBJECTIVES


def ax_cube(objective, n_trials, n_dim, with_count=False):
    global feval_count
    feval_count = 0

    def evaluation_func(prms):
        global feval_count
        feval_count += 1
        return objective([prms["u" + str(i)] for i in range(n_dim)])

    parameters = [{
        "name": "u" + str(i),
        "type": "range",
        "bounds": [0.0, 1.0],
    } for i in range(n_dim)]
    best_parameters, best_values, experiment, model = optimize(parameters=parameters,
                                                               evaluation_function=evaluation_func,
                                                               minimize=True,
                                                               total_trials=n_trials)
    return (best_values[0]['objective'], feval_count) if with_count else best_values[0]['objective']


@print_durations()
def demo():
    for objective in OBJECTIVES:
        print(ax_cube(objective, n_trials=50, n_dim=2, with_count=True))

if __name__ == '__main__':
    demo()

