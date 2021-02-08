import numpy as np
import random
from pprint import pprint
import traceback
from timemachines.optimizers.objectives import OBJECTIVES
from timemachines.optimizers.alloptimizers import OPTIMIZERS, optimizer_from_name
from ratings import elo_expected


def optimizer_elo_update(elo: dict, tol=0.001, initial_elo=1600):
    """ Create or update elo ratings for optimizers

          elo - Dictionary containing the 'state' (i.e. elo ratings and game counts)
          k   - Number of steps to look ahead
          tol - Objective function ratio that results in a tie being declared
          data_provider - A function returning y, t

        Speed is not taken into account
    """

    if not elo:
        # Initialize game counts and Elo ratings
        elo['name'] = [f.__name__ for f in OPTIMIZERS]
        elo['count'] = [0 for _ in OPTIMIZERS]
        elo['rating'] = [initial_elo for _ in OPTIMIZERS]
        elo['traceback'] = ['not yet run' for _ in OPTIMIZERS]
        elo['active'] = [True for _ in OPTIMIZERS]

    else:
        # Check for newcomers
        new_names = [f.__name__ for f in OPTIMIZERS if f.__name__ not in elo['name']]
        for new_name in new_names:
            elo['name'].append(new_name)
            elo['count'].append(0)
            elo['rating'].append(initial_elo)
            elo['traceback'].append('not yet run')
            elo['active'].append(True)

    n_optimizers = len(elo['name'])
    i1, i2 = np.random.choice(list(range(n_optimizers)), size=2, replace=False)
    optim_name_1, optim_name_2 = elo['name'][i1], elo['name'][i2]
    optims = list()
    for i,sn in zip([i1,i2],[optim_name_1,optim_name_2]):
        o = optimizer_from_name(sn)
        if o is not None:
            optims.append(o)
            elo['active'][i]=True
        else:
            elo['active'][i]=False

    if len(optims)==2:
        # Let's have at it !
        print(optims[0].__name__ +' vs. '+optims[1].__name__)
        n_dim = random.choice([1,2,3,5,8,13,21,34])
        n_trials = random.choice([10,20,30,50,80,130,210,340])
        objective = random.choice(OBJECTIVES)
        match_params = {'n_dim':n_dim,'n_trials':n_trials,'objective':objective.__name__}
        pprint(match_params)
        minima_found = list()
        trial_counts = list()

        for i, o,going_first in zip([i1,i2],optims,[True,False]):
            n_trials_to_use = n_trials if going_first else trial_counts[0]
            try:
                best_val, best_x, feval_count = o(objective, n_trials=n_trials_to_use, n_dim=n_dim, with_count=True)
                elo['traceback'][i] = 'passing'
                minima_found.append(best_val)
                trial_counts.append(feval_count)
            except Exception as e:
                elo['traceback'][i] = traceback.format_exc()
                trial_counts.append(None)

        if len(minima_found)==2:
            # Try to determine if one optimizer truly did a better job than the other.
            if trial_counts[0]>n_trials*1.5:
                message = 'Optimizer was naughty. Used '+str(trial_counts[0])+' evaluations when instructed to use '+str(n_trials_to_use)
                elo['traceback'][i1] = message
            elif trial_counts[1] > trial_counts[0]*1.2 or trial_counts[0]==n_trials and trial_counts[1] > trial_counts[0]*1.1:
                message = 'Optimizer was naughty. Used ' + str(trial_counts[1]) + ' evaluations when instructed to use ' + str(
                    n_trials_to_use)
                elo['traceback'][i2] = message
            elif minima_found[0] is None:
                message = 'Optimizer returned None on '+objective.__name__
                elo['traceback'][i1] = message
            elif minima_found[1] is None:
                message = 'Optimizer returned None on ' + objective.__name__
                elo['traceback'][i2] = message
            else:
                small = tol * (abs(minima_found[0]) + abs(minima_found[1]))  # Ties
                points = 1 if minima_found[0] < minima_found[1] - small else 0 if minima_found[1] < minima_found[0] - small else 0.5
                elo1, elo2 = elo['rating'][i1], elo['rating'][i2]
                min_games = min(elo['count'][i1],elo['count'][i2])
                K = 16 if min_games > 25 else 25
                elo['rating'][i1], elo['rating'][i2] = elo_update(elo1, elo2, points,K)
                elo['count'][i1] += 1
                elo['count'][i2] += 1
    else:
        match_params = dict()

    return elo, match_params


if __name__=='__main__':
    elo = {}
    while True:
        elo, _ =optimizer_elo_update(elo=elo)
        pprint(sorted(list( zip(elo['rating'],elo['name'])),reverse=True))