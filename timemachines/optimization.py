from timemachines.conventions import to_space, Y_TYPE, from_space, dimension
from timemachines.evaluation import evaluate_energy, evaluate_mean_squared_error
from timemachines.optimizers.alloptimizers import OPTIMIZERS, shgo_cube, optuna_cube
from timemachines.skaters.pmd import pmd_auto


def optimize(f, ys:[Y_TYPE],
             evaluator,
             optimizer,
             n_trials,
             n_dim,        # Which dimension to search in
             **kwargs)->(float, float):
    """ Returns best r """

    def objective(u:[float]):
        r = from_space(u)
        return evaluator(f=f,ys=ys,r=r,**kwargs)

    a_test = objective(u=[0.5]*n_dim)  # Fail fast with easier trace TODO: remove
    return optimizer(objective, n_trials=n_trials, n_dim=n_dim, with_count=False)



if __name__=='__main__':
    from timemachines.synthetic import brownian_with_exogenous
    best_val, best_x = optimize(f=pmd_auto,ys=brownian_with_exogenous(n=60),
                      n_trials=5, n_dim=3, n_burn=20, optimizer=optuna_cube, evaluator=evaluate_mean_squared_error)
    best_r = from_space(best_x)
    print("Best hyper-param is "+str(best_r))
    from timemachines.skaters.pmd import pmd_hyperparams
    s = pmd_hyperparams(s=dict(),r=best_r)
    from pprint import pprint
    pprint(s)


