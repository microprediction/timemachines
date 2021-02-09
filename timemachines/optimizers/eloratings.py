import numpy as np
import random
from pprint import pprint
import traceback
from timemachines.objectives.classic import CLASSIC_OBJECTIVES
from timemachines.optimizers.alloptimizers import OPTIMIZERS
from timemachines.common.eloratings import elo_update


N_DIM_CHOICES = [1, 2, 3, 5, 8]
N_TRIALS_CHOICES = [10,20,30]

OPTIMIZER_F_FACTOR=1000
OPTIMIZER_K_FACTOR=25


def optimizer_game(white, black, n_dim, n_trials, objective, tol=0.001):
    """
    :param white:   optimizer
    :param black:   optimizer
    :param n_dim:
    :param n_trials:
    :param objective:
    :return:  dict
    """

    game_result = {'n_dim': n_dim, 'n_trials': n_trials, 'objective': objective.__name__,'white':white, 'black':black,
             'traceback':['passing','passing'],'best_val':[None, None],'best_x':[None,None],'feval_count':[None,None]}

    minima_found = list()
    trial_counts = list()

    for j, optimizer, going_first in zip([0,1], [white, black], [True, False]):
        n_trials_to_use = n_trials if going_first else trial_counts[0]
        try:
            best_val, best_x, feval_count = optimizer(objective, n_trials=n_trials_to_use, n_dim=n_dim, with_count=True)
            game_result['traceback'][j] = 'passing'
            minima_found.append(best_val)
            trial_counts.append(feval_count)
            game_result['best_val'][j]=best_val
            game_result['best_x'][j]=best_x
            game_result['feval_count'][j]=feval_count
        except Exception as e:
            game_result['traceback'][j] = traceback.format_exc()
            trial_counts.append(None)

    game_result['completed']=False
    if len(minima_found) == 2:
        # Try to determine if one optimizer truly did a better job than the other.
        if trial_counts[0] > n_trials * 1.5:
            message = 'Optimizer was naughty. Used ' + str(
                trial_counts[0]) + ' evaluations when instructed to use ' + str(n_trials_to_use)
            game_result['traceback'][j] = message
        elif trial_counts[1] > trial_counts[0] * 1.4 or trial_counts[0] == n_trials and trial_counts[1] > trial_counts[
            0] * 1.2:
            message = 'Optimizer was naughty. Used ' + str(
                trial_counts[1]) + ' evaluations when instructed to use ' + str(
                n_trials_to_use)
            game_result['traceback'][j] = message
        elif minima_found[0] is None:
            message = 'Optimizer returned None on ' + objective.__name__
            game_result['traceback'][j] = message
        elif minima_found[1] is None:
            message = 'Optimizer returned None on ' + objective.__name__
            game_result['traceback'][j] = message
        else:
            small = tol * (abs(minima_found[0]) + abs(minima_found[1]))  # Ties
            points = 1 if minima_found[0] < minima_found[1] - small else 0 if minima_found[1] < minima_found[
                0] - small else 0.5
            game_result['white_points']=points
            game_result['completed']=True
            game_result['winner']= white.__name__ if points>0.75 else black.__name__ if points<0.25 else 'draw'
    else:
        game_result['winner']='incomplete'
        game_result['white_points']=None
    return game_result


def random_optimizer_game(optimizers=None, objectives=None, n_dim_choices:[int]=None, n_trials_choices:[int]=None, tol=0.001):
    if n_dim_choices is None:
        n_dim_choices = N_DIM_CHOICES

    if n_trials_choices is None:
        n_trials_choices = N_TRIALS_CHOICES

    if objectives is None:
        objectives = CLASSIC_OBJECTIVES

    if optimizers is None:
        optimizers = OPTIMIZERS

    n_dim = random.choice(n_dim_choices)
    n_trials = random.choice(n_trials_choices)
    objective = random.choice(objectives)
    white, black = np.random.choice(optimizers, size=2, replace=False)
    game_result = optimizer_game(white=white, black=black, n_dim=n_dim, n_trials=n_trials, objective=objective, tol=tol)
    return game_result


def optimizer_population_elo_update(optimizers, game_result:dict, elo: dict, initial_elo=1600):
    """ Create or update elo ratings for optimizers

          optimizers - List of optimizers that were considered
          game_result - Produced by optimizer_game
          elo   - Dictionary containing the 'state' of the population (i.e. elo ratings and game counts)
          tol   - Objective function ratio that results in a tie being declared

        Chooses random objective function, random dimensions and random number of trials
        Speed is not taken into account
    """

    if not elo:
        # Initialize game counts and Elo ratings
        elo['name'] = [f.__name__ for f in optimizers]
        elo['count'] = [0 for _ in optimizers]
        elo['rating'] = [initial_elo for _ in optimizers]
        elo['traceback'] = ['not yet run' for _ in optimizers]
        elo['active'] = [True for _ in optimizers]

    else:
        # Check for newcomers
        new_names = [f.__name__ for f in optimizers if f.__name__ not in elo['name']]
        for new_name in new_names:
            elo['name'].append(new_name)
            elo['count'].append(0)
            elo['rating'].append(initial_elo)
            elo['traceback'].append('not yet run')
            elo['active'].append(True)
    # Who is active?
    optimizer_names = [ o.__name__ for o in optimizers ]
    elo['active'] = [ name_ in optimizer_names for name_ in elo['name'] ]

    # Peg rating of randomized algorithms to 1600, say.
    elo['rating'] = [ r if 'random' not in name_ else initial_elo for r,name_ in zip(elo['rating'],elo['name']) ]

    # Process results of match
    white_name = game_result['white'].__name__
    black_name = game_result['black'].__name__
    white_ndx = elo['name'].index(white_name)
    black_ndx = elo['name'].index(black_name)
    elo['count'][white_ndx]+=1
    elo['count'][black_ndx]+=1
    if game_result['completed']:
        points = game_result['points']
        winner = white_name if points>0.75 else black_name if points<0.25 else 'draw'
        print('>>>> ' + winner)
        white_elo, black_elo = elo['rating'][white_ndx], elo['rating'][black_ndx]
        min_games = min(elo['count'][white_ndx],elo['count'][black_ndx])
        k = OPTIMIZER_K_FACTOR/2.0 if min_games > 10 else OPTIMIZER_K_FACTOR
        new_white_elo, new_black_elo = elo_update(white_elo=white_elo, black_elo=black_elo,points=points,k=k,f=OPTIMIZER_F_FACTOR)
        if elo['count'][black_ndx]>3:
            elo['rating'][white_ndx] = new_white_elo
        if elo['count'][white_ndx]>3:
            elo['rating'][white_ndx] = new_black_elo
    else:
        print('>>>> incomplete ')

    return elo

def demo_optimizer_elo():
    # Run this to generate Elo ratings that will update for as long as you have the patience.
    elo = {}
    while True:
        elo = optimizer_population_elo_update(elo=elo)
        print(' ')
        pprint(sorted(list(zip(elo['rating'], elo['name'])), reverse=True))
        print(' ')

if __name__=='__main__':
    demo_optimizer_elo()