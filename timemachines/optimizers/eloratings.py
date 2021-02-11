import numpy as np
import random
from pprint import pprint
import traceback
from timemachines.objectives.classic import CLASSIC_OBJECTIVES
from timemachines.optimizers.alloptimizers import OPTIMIZERS
from timemachines.common.eloratings import elo_update


N_DIM_CHOICES = [1, 2, 3, 5, 8]
N_TRIALS_CHOICES = [ 130, 210, 340]

OPTIMIZER_F_FACTOR = 1000
OPTIMIZER_K_FACTOR = 60
N_PROVISIONAL = 0   # Number of games for which player is considered provisional
N_ATTEMPTS_WHITE   = 3
N_ATTEMPTS_BLACK   = 6


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
             'traceback':['passing','passing'],'best_val':[None, None],'best_x':[None,None],'feval_count':[None,None],
                   'n_trials_instructed':[None, None],'passing':[None, None],
                   'completed':False}

    # White to play..
    n_white_trials = n_trials
    n_white_attempts = 0
    while True:
        try:
            white_best_val, white_best_x, white_feval_count = white(objective, n_trials=n_white_trials, n_dim=n_dim, with_count=True)
            white_passing, white_traceback = white_best_val is not None, 'passing'
        except Exception as e:
            white_traceback = traceback.format_exc()
            white_passing, white_best_x, white_best_val, white_feval_count = False, None, None, None
        if not white_passing or (white_feval_count<=n_trials) or (n_white_attempts>N_ATTEMPTS_WHITE):
            break
        else:
            n_white_attempts+=1
            print('Playing white,'+white.__name__ + ' attempt ' + str(n_white_attempts + 1) + ' after instruction to use ' + str(n_white_trials) + ' resulted in '+str(white_feval_count)+' evaluations.')
            n_white_trials = int(0.7*n_white_trials)
    game_result['best_val'][0] = white_best_val
    game_result['passing'][0] = white_passing
    white_success = white_passing and white_feval_count <= n_trials
    if white_passing and n_white_trials > n_trials:
        white_traceback = 'White took ' + str(white_feval_count) + ' function evals when instructed to use ' + str(
            n_white_trials)
    game_result['traceback'][0] = white_traceback
    if white_passing:
        game_result['n_trials_instructed'][0] = n_white_trials
        game_result['feval_count'][0] = white_feval_count
        game_result['best_val'][0] = white_best_val
        game_result['best_x'][0] = white_best_x
    else:
        pass

    # Black to play
    if white_success:
        n_black_trials = white_feval_count  # <-- Tries to match the number of white actual evaluations
        n_black_attempts = 0
        while True:
            try:
                black_best_val, black_best_x, black_feval_count = black(objective, n_trials=n_black_trials,
                                                                    n_dim=n_dim,
                                                                    with_count=True)
                black_traceback, black_passing = 'passing', black_best_val is not None
            except Exception as e:
                black_traceback = traceback.format_exc()
                black_passing, black_best_x, black_best_val, black_feval_count = False, None, None, None

            if not black_passing or (black_feval_count <= 1.1*white_feval_count) or (n_black_attempts > N_ATTEMPTS_BLACK):
                break
            else:
                n_black_attempts += 1
                print('Playing black, '+black.__name__ + ' attempt ' + str(n_black_attempts + 1) + ' after instruction to use ' + str(
                    n_black_trials) + ' resulted in ' + str(black_feval_count) + ' evaluations.')
                n_black_trials = int(0.85 * n_black_trials)
        black_success = black_passing and black_feval_count <= 1.1*white_feval_count
        game_result['n_trials_instructed'][1] = n_black_trials
        game_result['feval_count'][1] = black_feval_count
        game_result['best_val'][1] = black_best_val
        if black_passing and n_black_trials > n_trials:
            black_traceback = 'Black took '+str(black_feval_count)+' function evals when instructed to use '+str(n_black_trials)
        game_result['traceback'][1] = black_traceback
        game_result['passing'][1] = black_passing
        if black_passing:
            game_result['best_val'][1] = black_best_val
            game_result['best_x'][1] = black_best_x
            game_result['n_trials_instructed'][1] = n_black_trials
            game_result['feval_count'][1] = black_feval_count

    # Now that White and Black have both played...
    if white_success and black_success:
        game_result['completed']=True
        small = tol * (abs(white_best_val) + abs(black_best_val))  # Ties
        points = 1. if white_best_val < black_best_val - small else 0. if black_best_val < white_best_val - small else 0.5
        game_result['points'] = points
        game_result['winner'] = white.__name__.replace('_cube','') if points > 0.75 else black.__name__ if points < 0.25 else 'draw'
        game_result['loser'] = black.__name__.replace('_cube',
                                                   '') if points > 0.75 else white.__name__ if points < 0.25 else 'draw'
        if game_result['winner']!='draw':
            print(game_result['winner']+' beats '+game_result['loser'])
        else:
            print(black.__name__ + ' holds '+white.__name__+' to a draw.')
    else:
        game_result['winner'] = 'incomplete'
        game_result['points'] = None

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

    # Peg to random sampling
    if True:
        elo['rating'] = [ initial_elo if 'optuna_random' in name_ else r for r,name_ in zip(elo['rating'],elo['name']) ]

    # Process results of match
    white_name = game_result['white'].__name__
    black_name = game_result['black'].__name__
    white_ndx = elo['name'].index(white_name)
    black_ndx = elo['name'].index(black_name)
    elo['traceback'][white_ndx]=game_result['traceback'][0]
    elo['traceback'][black_ndx]=game_result['traceback'][1]
    elo['count'][black_ndx]+=1
    if game_result['completed']:
        points = game_result['points']
        print('>>>> ' + game_result['winner'])
        white_elo, black_elo = elo['rating'][white_ndx], elo['rating'][black_ndx]
        min_games = min(elo['count'][white_ndx],elo['count'][black_ndx])
        k = OPTIMIZER_K_FACTOR/2.0 if min_games > 10 else OPTIMIZER_K_FACTOR
        new_white_elo, new_black_elo = elo_update(white_elo=white_elo, black_elo=black_elo,points=points,k=k,f=OPTIMIZER_F_FACTOR)
        # Don't allow players with provisional ratings to impact other's.
        if elo['count'][black_ndx]>=N_PROVISIONAL:
            elo['rating'][white_ndx] = new_white_elo
        if elo['count'][white_ndx]>N_PROVISIONAL:
            elo['rating'][black_ndx] = new_black_elo
    else:
        print('>>>> incomplete ')

    return elo

def demo_optimizer_elo():
    # Run this to generate Elo ratings that will update for as long as you have the patience.
    elo = {}
    while True:
        game_result = random_optimizer_game(optimizers=OPTIMIZERS, objectives=CLASSIC_OBJECTIVES,
                                            n_dim_choices=N_DIM_CHOICES, n_trials_choices=N_TRIALS_CHOICES, tol=0.001)
        print(' Game...')
        pprint(game_result)

        elo = optimizer_population_elo_update(optimizers=OPTIMIZERS,elo=elo,game_result=game_result)
        if random.choice(list(range(5)))==1:
            print(' ')
            pprint(sorted(list(zip(elo['rating'], elo['name'])), reverse=True))
            print(' ')

if __name__=='__main__':
    demo_optimizer_elo()