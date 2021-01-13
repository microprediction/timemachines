from timemachines.conventions import to_space, Y_TYPE, from_space, dimension
from timemachines.evaluation import evaluate_energy, evaluate_mean_squared_error
from timemachines.optimizers.compendium import OPTIMIZERS, shgo_cube


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
    from timemachines.skaters.pmd import pmd_auto
    from timemachines.synthetic import brownian_with_exogenous
    r_star = optimize(f=pmd_auto,ys=brownian_with_exogenous(n=500),n_trials=50)
    print("Best hyper-param is "+str(r_star))


