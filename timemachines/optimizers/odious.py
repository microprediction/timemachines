from pprint import pprint
from timemachines.optimizers.alloptimizers import OPTIMIZERS
from timemachines.optimizers.objectives import OBJECTIVES
import random
from typing import List
import numpy as np
from pprint import pprint
from ratings.elo import elo_expected


def optimizer_name(solver):
    return solver.__name__.replace('_cube','')


def objective_name(objective):
    return objective.__name__.replace('_cube','')


def debug():
    for n_trials in [2,10,40]:
        for n_dim in [2,3]:
            for optimizer in OPTIMIZERS:
                for objective in OBJECTIVES:
                    try:
                        optimizer(objective, n_trials=n_trials,n_dim=n_dim,with_count=True)
                    except Exception as e:
                        print(e)
                        print(optimizer_name(optimizer) + ' fails on ' + objective.__name__ +
                              ' in dimension '+str(n_dim)+' when n_trials='+str(n_trials))


def comparison():
    n_trials,n_dim = 300, 5
    comparison = sorted([(objective.__name__,optimizer(objective, n_trials=n_trials, n_dim=n_dim, with_count=True),
                          optimizer_name(optimizer))
                         for optimizer in OPTIMIZERS for objective in OBJECTIVES])
    pprint(comparison)


def random_matchup(optimizers:List, objectives:List) -> dict:
    optim1, optim2 = np.random.choice( optimizers, 2, replace=False )
    objective = np.random.choice( objectives, 1 )[0]
    return matchup(optim1,optim2,objective)


def matchup( white, black, objective )-> dict:
    """ Compare two optimizers, making an effort to equalize function evaluations
    :param white:   optimizer
    :param black:   optimizer
    :param objective:
    :return: match result dictionary
    """
    n_trials = np.random.choice( [50,100,200,400,800,1600] )
    n_dim = np.random.choice( [1,2,4,6,8,10,12,16] )
    best_val_white, best_x_white, count_white = white(objective, n_trials=n_trials, n_dim=n_dim, with_count=True)
    best_val_black, best_x_black, count_black = black(objective, n_trials=int(0.8*count_white),  n_dim=n_dim, with_count=True)
    valid_white, valid_black = count_white<2*n_trials, count_black <= n_trials

    tol = 1e-3*(abs(best_val_white)+abs(best_val_black))
    if valid_white and valid_black:
        if best_val_white < best_val_black - tol:
            points = 1.0
        elif best_val_black < best_val_white - tol:
            points = 0.0
        else:
            points = 0.5
    else:
        points = None

    return {'white':optimizer_name(white),'black':optimizer_name(black),'points':points,'n_dim':n_dim,
            'n_trials':n_trials,'white_best_val':best_val_white,'black_best_val':best_val_black,
            'white_trails':count_white,'black_trials':count_black}




if __name__=='__main__':
    if False:
        comparison()
    res = random_matchup(optimizers=OPTIMIZERS,objectives=OBJECTIVES)
    pprint(res)



