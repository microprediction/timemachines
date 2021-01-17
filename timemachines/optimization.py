from timemachines.conventions import to_space, Y_TYPE, from_space, dimension
from timemachines.evaluation import evaluate_energy, evaluate_mean_squared_error
from timemachines.optimizers.alloptimizers import OPTIMIZERS
from timemachines.skaters.pmd import pmd_auto
import traceback

def optimize(f, ys:[Y_TYPE],
             evaluator,
             optimizer,
             n_trials,
             n_dim,        # Which dimension to search in
             with_count=False,
             **kwargs)->(float, float):
    """ Returns best r """

    def objective(u:[float]):
        r = from_space(u)
        return evaluator(f=f,ys=ys,r=r,**kwargs)

    a_test = objective(u=[0.5]*n_dim)  # Fail fast with easier trace TODO: remove
    return optimizer(objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count)



if __name__=='__main__':
    from timemachines.synthetic import brownian_with_exogenous
    from timemachines.evaluation import EVALUATORS
    broken = list()
    for optimizer in OPTIMIZERS:
        print(' ')
        print(optimizer.__name__)
        for evaluator in EVALUATORS:
            try:
                print( evaluator.__name__, optimize(f=pmd_auto,ys=brownian_with_exogenous(n=120),
                      n_trials=5, n_dim=3, n_burn=20, optimizer=optimizer,
                      evaluator=evaluator, with_count=True))
            except Exception as e:
                traceback.print_tb()
                broken.append( (optimizer.__name__,evaluator.__name__))
    print(' ')
    print('Broken : ')
    print(broken)

