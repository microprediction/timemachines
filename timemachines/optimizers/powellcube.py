from scipy.optimize import minimize

global feval_count
feval_count=0


def powell_cube(objective, n_trials, n_dim, with_count=False):
    bounds = [(0,1) ]*n_dim

    global feval_count
    feval_count = 0

    def _objective(x):
        global feval_count
        feval_count +=1
        return objective(list(x))

    result = minimize(_objective, x0=[0]*n_dim, method='powell',bounds=bounds, options={'maxfev':n_trials,'maxiter':n_trials})
    return (_objective(result.x), feval_count) if with_count else _objective(result.x)


if __name__ == '__main__':
    from timemachines.optimizers.objectives import OBJECTIVES
    for objective in OBJECTIVES:
        print(powell_cube(objective, n_trials=5, n_dim=50, with_count=True))
