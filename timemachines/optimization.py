from timemachines.conventions import to_space, Y_TYPE, from_space, dimension
from timemachines.evaluation import evaluate_energy, evaluate_mean_squared_error
from timemachines.optimizers.compendium import OPTIMIZERS, shgo_cube, optuna_cube


def optimize(f, ys:[Y_TYPE],
             evaluator=evaluate_mean_squared_error,
             optimizer=shgo_cube,
             n_trials=100,
             **kwargs)->float:
    """ Returns best r """

    def objective(u:[float]):
        r = from_space(u)
        return evaluator(f=f,ys=ys,r=r,**kwargs)

    return optimizer(objective, n_trials=n_trials, n_dim=dimension(ys[0]))


if __name__=='__main__':
    if False:
        from timemachines.synthetic import brownian_with_exogenous
        r_star = optimize(f=f,ys=brownian_with_exogenous(n=300),
                          n_trials=25, optimizer=optuna_cube)
        print("Best hyper-param is "+str(r_star))
    else:
        r_star = 0.468933170971917
        from timemachines.skaters.pmd import pmd_hyperparams
        s = pmd_hyperparams(s=dict(),r=r_star)
        from pprint import pprint
        pprint(s)


