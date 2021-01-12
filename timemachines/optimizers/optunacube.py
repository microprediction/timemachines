import optuna
from optuna.logging import CRITICAL
from timemachines.optimizers.objectives import OBJECTIVES


def optuna_cube(objective, n_trials,n_dim, with_count=False):

    global feval_count
    feval_count = 0

    def cube_objective(trial):
        global feval_count
        us = [ trial.suggest_float('u'+str(i),0,1) for i in range(n_dim)]
        feval_count += 1
        return objective(us)

    optuna.logging.set_verbosity(CRITICAL)
    study = optuna.create_study()
    study.optimize(cube_objective,n_trials=n_trials)
    if with_count:
        return study.best_value, feval_count
    else:
        return study.best_value


if __name__=='__main__':
    for objective in OBJECTIVES:
        print((objective.__name__,optuna_cube(objective, n_trials=100, n_dim=6, with_count=True)))
