from timemachines.conventions import to_space, Y_TYPE, from_space, dimension
from timemachines.evaluation import evaluate_energy, evaluate_mean_squared_error
from timemachines.optimizers.alloptimizers import OPTIMIZERS, shgo_cube, optuna_cube
from timemachines.skaters.pmd import pmd_auto


def optimize(f, ys:[Y_TYPE],
             evaluator=evaluate_mean_squared_error,
             optimizer=shgo_cube,
             n_trials=100,
             **kwargs)->(float, float):
    """ Returns best r """

    def objective(u:[float]):
        r = from_space(u)
        return evaluator(f=f,ys=ys,r=r,**kwargs)

    r_star = optimizer(objective, n_trials=n_trials, n_dim=dimension(ys[0]))
    evaluation = objective(r_star)
    return r_star, evaluation


if __name__=='__main__':
    # Example that could take a while
    from timemachines.synthetic import brownian_with_exogenous
    r_star, evaluation = optimize(f=pmd_auto,ys=brownian_with_exogenous(n=300),
                      n_trials=25, optimizer=optuna_cube)
    print("Best hyper-param is "+str(r_star))
    from timemachines.skaters.pmd import pmd_hyperparams
    s = pmd_hyperparams(s=dict(),r=r_star)
    from pprint import pprint
    pprint(s)


