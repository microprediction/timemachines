import numpy as np
from timemachines.skatertools.comparison.eloformulas import elo_update
import time
from pprint import pprint

SLOW_SKATER_KEYWORDS = ['divine','fbprophet','arma','tsa_p3','bats_damped','tsa_aggressive','tsa_balanced','tsa_precision']

try:
    from microprediction import MicroReader
    have_micro = True
except ImportError:
    have_micro = False

if have_micro:
    from timemachines.skatertools.data.live import random_regular_data as DEFAULT_DATA_SOURCE
else:
    DEFAULT_DATA_SOURCE = None

SKATER_F_FACTOR = 1000  # The scale factor for ratings. In chess this is set to 400.
# As the matchup is considered more fluky than a single game of chess, a higher value makes sense.
SKATER_K_FACTOR = 200  # The Elo update factor (maximum rating gain)


def skater_elo_multi_update(elo: dict, k, evaluator=None, n_burn=400, tol=0.01, initial_elo=1600, data_source=None):
    """ Create or update elo ratings by running several algos at once and using Elo update n-1 times

              elo - Dictionary containing the 'state' (i.e. elo ratings and game counts)
              k   - Number of steps to look ahead
              tol - Error ratio that results in a tie being declared
              data_provider - A function taking n_obs and returning y, t
    """
    # Lazy import because networked skaters shouldn't be initialized too early
    from timemachines.skaters.pypi import pypi_from_name
    from timemachines.skaters.allskaters import SKATERS, skater_from_name  # Only those with no hyper-params

    if data_source is None:
        data_source = DEFAULT_DATA_SOURCE
        assert data_source is not None, "If microprediction is not installed you must supply a different data_source function that returns y, t "

    if not elo:
       _init_elo(elo=elo,SKATERS=SKATERS,initial_elo=initial_elo)
    else:
        # New fields ... one-off fix
        if not 'pypi' in elo:
            elo['pypi'] = [ pypi_from_name(nm) for nm in elo['name']]
        if not 'seconds' in elo:
            elo['seconds'] = [-1 for nm in elo['name']]
        elo['pypi'] = [pypi_from_name(nm) for nm in elo['name']]

        # Check for newcomers
        elo = _newcomers(elo=elo,SKATERS=SKATERS,initial_elo=initial_elo)
        new_names = [f.__name__ for f in SKATERS if f.__name__ not in elo['name']]

    elo, evaluator = _set_evaluator(elo)

    # Choose several skaters
    total_seconds = 0
    chosen = list()
    n_skaters = len(elo['name'])
    candidates = np.random.choice(list(range(n_skaters)), size=10, replace=False)
    for c in candidates:
        snds = elo['seconds'][c]
        cnt = elo['count'][c]
        if cnt<10:
            snds = snds/3
        if cnt<25:
            snds = snds/2
        if snds<0:
            snds = 45
        if (snds+total_seconds<60) or ((len(chosen)>2) and (np.random.rand()<(300/(2+snds)) )):
            total_seconds += snds
            chosen.append(c)
        if total_seconds>60:
            break

    fs = list()
    chosen_and_imported = list()
    for c in chosen:
        sn = elo['name'][c]
        try:
            f = skater_from_name(sn)
            assert f is not None, 'probably just not used anymore'
            elo['active'][c] = True
            fs.append(f)
            chosen_and_imported.append(c)
        except Exception as e:
            elo['active'][c] = False

    got_data = False
    while not got_data:
        try:
            y, t = data_source(n_obs=n_burn + 50)
            got_data = True
        except:
            time.sleep(5)
    scores = list()
    ran_okay = list()
    ran_okay_names = list()
    failed_names = list()
    print('  imported successfully ...')
    pprint(chosen_and_imported)
    for c, f in zip(chosen_and_imported, fs):
        import traceback
        try:
            st = time.time()
            score = evaluator(f=f, y=y, k=k, a=None, t=t, e_fit=15, e_nofit=-1, n_test=50, fit_frequency=100)
            elo['seconds'][c] = time.time() - st
            elo['traceback'][c] = 'passing'
            scores.append(score)
            ran_okay.append(c)
            ran_okay_names.append(f.__name__)
        except Exception as e:
            elo['traceback'][c] = traceback.format_exc()
            elo['seconds'][c] = st - time.time()  # Time taken to fail, as -ve number
            failed_names.append(f.__name__)

    if any(failed_names):
        print('  failing ...')
        pprint(failed_names)

    print('  leaderboard ...')
    leaderboard = sorted([(s,c,n) for s,c,n in zip(scores,ran_okay, ran_okay_names)])
    pprint(leaderboard)
    if len(leaderboard)>=2:
        for j, (winner_score,winner, winner_name) in enumerate(leaderboard[:-1]):
            loser = leaderboard[j+1][1]
            loser_score = leaderboard[j+1][0]
            small = tol * (abs(winner_score) + abs(loser_score))  # Ties
            min_games = min(elo['count'][winner], elo['count'][loser])
            K = SKATER_K_FACTOR / 2.0 if min_games > 25 else SKATER_K_FACTOR  # The Elo update scaling parameter
            winner_prior_elo, loser_prior_elo = elo['rating'][winner], elo['rating'][loser]
            points = 1 if winner_score < loser_score - small else 0.5
            elo['rating'][winner], elo['rating'][loser] = elo_update(winner_prior_elo, loser_prior_elo, points, k=K, f=SKATER_F_FACTOR)
            elo['count'][winner] += 1
            elo['count'][loser] += 1
    return elo


def _init_elo(elo, SKATERS, initial_elo):
    from timemachines.skaters.pypi import pypi_from_name
    # Initialize game counts and Elo ratings
    elo['name'] = [f.__name__ for f in SKATERS]
    elo['count'] = [0 for _ in SKATERS]
    elo['rating'] = [initial_elo for _ in SKATERS]
    elo['traceback'] = ['not yet run' for _ in SKATERS]
    elo['active'] = [True for _ in SKATERS]
    elo['pypi'] = [pypi_from_name(nm) for nm in elo['name']]
    return elo

def _newcomers(elo, SKATERS,initial_elo):
    # Check for newcomers
    from timemachines.skaters.pypi import pypi_from_name
    new_names = [f.__name__ for f in SKATERS if f.__name__ not in elo['name']]
    for new_name in new_names:
        elo['name'].append(new_name)
        elo['count'].append(0)
        elo['rating'].append(initial_elo)
        elo['traceback'].append('not yet run')
        elo['active'].append(True)
        elo['seconds'].append(-1)
        elo['pypi'].append(pypi_from_name(new_name))
    return elo

def _set_evaluator(elo):
    from timemachines.skatertools.evaluation.evaluators import evaluate_mean_squared_error_with_sporadic_fit, \
        evaluator_from_name
    evaluator = evaluate_mean_squared_error_with_sporadic_fit
    if evaluator is None:
        if elo.get('evaluator'):
            eval_name = elo['evaluator']
            try:
                evaluator = evaluator_from_name(eval_name)
            except:
                print('Could not retrieve ' + eval_name + ' so reverting.')
                evaluator = evaluate_mean_squared_error_with_sporadic_fit
        else:
            evaluator = evaluate_mean_squared_error_with_sporadic_fit
        print('Evaluating using ' + str(evaluator.__name__))
    elo['evaluator'] = evaluator.__name__
    return elo, evaluator


def skater_elo_update(elo: dict, k, evaluator=None, n_burn=400, tol=0.01, initial_elo=1600, data_source=None):
    """ Create or update elo ratings by performing a random matchup on univariate live data

          elo - Dictionary containing the 'state' (i.e. elo ratings and game counts)
          k   - Number of steps to look ahead
          tol - Error ratio that results in a tie being declared
          data_provider - A function taking n_obs and returning y, t

        Speed is *not* taken into account, yet.
    """
    # Lazy import because networked skaters shouldn't be initialized too early
    from timemachines.skaters.pypi import pypi_from_name
    from timemachines.skaters.allskaters import SKATERS, \
        skater_from_name  # Only those with no hyper-params

    if data_source is None:
        data_source = DEFAULT_DATA_SOURCE
        assert data_source is not None, "If microprediction is not installed you must supply a different data_source function that returns y, t "

    if not elo:
       _init_elo(elo=elo,SKATERS=SKATERS,initial_elo=initial_elo)
    else:
        # New fields ... one-off fix
        if not 'pypi' in elo:
            elo['pypi'] = [ pypi_from_name(nm) for nm in elo['name']]
        if not 'seconds' in elo:
            elo['seconds'] = [-1 for nm in elo['name']]
        elo['pypi'] = [pypi_from_name(nm) for nm in elo['name']]

        # Check for newcomers
        elo = _newcomers(elo=elo,SKATERS=SKATERS,initial_elo=initial_elo)
        new_names = [f.__name__ for f in SKATERS if f.__name__ not in elo['name']]

    elo, evaluator = _set_evaluator(elo)

    # Choose two random skaters, but avoid slow ones once
    n_skaters = len(elo['name'])
    i1, i2 = np.random.choice(list(range(n_skaters)), size=2, replace=False)
    skater1, skater2 = elo['name'][i1], elo['name'][i2]
    for _ in range(2):
        if any([ slow in skater1 for slow in SLOW_SKATER_KEYWORDS ]) or any([ slow in skater2 for slow in SLOW_SKATER_KEYWORDS ]):
            i1, i2 = np.random.choice(list(range(n_skaters)), size=2, replace=False)
            skater1, skater2 = elo['name'][i1], elo['name'][i2]
    if any([ slow in skater1 for slow in SLOW_SKATER_KEYWORDS ]) and any([ slow in skater2 for slow in SLOW_SKATER_KEYWORDS ]):
        i1, i2 = np.random.choice(list(range(n_skaters)), size=2, replace=False)
        skater1, skater2 = elo['name'][i1], elo['name'][i2]


    fs = list()
    for i, sn in zip([i1, i2], [skater1, skater2]):
        try:
            f = skater_from_name(sn)
            assert f is not None, 'probably just not used anymore'
            elo['active'][i] = True
            fs.append(f)
        except Exception as e:
            elo['active'][i] = False

    if len(fs) == 2:
        # a pitched paddle battle in a bottle
        print(fs[0].__name__ + ' vs. ' + fs[1].__name__)
        y, t = data_source(n_obs=n_burn + 50)
        scores = list()
        for i, f in zip([i1, i2], fs):
            import traceback
            try:
                st = time.time()
                score = evaluator(f=f, y=y, k=k, a=None, t=t, e_fit=15, e_nofit=-1, n_test=50, fit_frequency=100)
                elo['seconds'][i] = time.time()-st
                elo['traceback'][i] = 'passing'
                scores.append(score)
            except Exception as e:
                elo['traceback'][i] = traceback.format_exc()
                elo['seconds'][i] = st-time.time() # Time taken to fail, as -ve number

        if len(scores) == 2:
            small = tol * (abs(scores[0]) + abs(scores[1]))  # Ties
            points = 1 if scores[0] < scores[1] - small else 0 if scores[1] < scores[0] - small else 0.5
            elo1, elo2 = elo['rating'][i1], elo['rating'][i2]
            min_games = min(elo['count'][i1], elo['count'][i2])
            K = SKATER_K_FACTOR / 2.0 if min_games > 25 else SKATER_K_FACTOR  # The Elo update scaling parameter
            elo['rating'][i1], elo['rating'][i2] = elo_update(elo1, elo2, points, k=K, f=SKATER_F_FACTOR)
            elo['count'][i1] += 1
            elo['count'][i2] += 1
            the_winner = skater1 if points > 0.5 else skater2 if points<0.5 else 'nobody'
            print('     won by '+the_winner)

    return elo
