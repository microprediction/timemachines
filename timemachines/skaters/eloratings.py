from timemachines.skaters.allskaters import SKATERS, skater_from_name  # Only those with no hyper-params
from timemachines.skaters.evaluation import evaluate_mean_squared_error, evaluator_from_name
import numpy as np
from timemachines.data.live import random_regular_data


def skater_elo_update(elo: dict, k, evaluator=None, n_burn=400, tol=0.01, initial_elo=1600):
    """ Create or update elo ratings by performing a random matchup on univariate live data

          elo - Dictionary containing the 'state' (i.e. elo ratings and game counts)
          k   - Number of steps to look ahead
          tol - Error ratio that results in a tie being declared

        Speed is not taken into account
    """

    if not elo:
        # Initialize game counts and Elo ratings
        elo['name'] = [f.__name__ for f in SKATERS]
        elo['count'] = [0 for _ in SKATERS]
        elo['rating'] = [initial_elo for _ in SKATERS]
        elo['traceback'] = ['not yet run' for _ in SKATERS]
        elo['active'] = [True for _ in SKATERS]

    else:
        # Check for newcomers
        new_names = [f.__name__ for f in SKATERS if f.__name__ not in elo['name']]
        for new_name in new_names:
            elo['name'].append(new_name)
            elo['count'].append(0)
            elo['rating'].append(initial_elo)
            elo['traceback'].append('not yet run')
            elo['active'].append(True)

    if evaluator is None:
        if elo.get('evaluator'):
             evaluator = evaluator_from_name(elo.get('evaluator'))
        else:
            evaluator = evaluate_mean_squared_error
        elo['evaluator'] = evaluator.__name__

    n_skaters = len(elo['name'])
    i1, i2 = np.random.choice(list(range(n_skaters)), size=2, replace=False)
    skater1, skater2 = elo['name'][i1], elo['name'][i2]
    fs = list()
    for i,sn in zip([i1,i2],[skater1,skater2]):
        try:
            f = skater_from_name(sn)
            elo['active'][i] = True
            fs.append(f)
        except Exception as e:
            elo['active'][i]=False

    if len(fs)==2:
        # paddle battle in a bottle
        print(fs[0].__name__ +' vs. '+fs[1].__name__)
        y, t = random_regular_data(n_obs=500)
        scores = list()
        for i, f in zip([i1,i2],fs):
            import traceback
            try:
                score = evaluator(f=f, y=y, k=k, a=None, t=t, e=None, n_burn=n_burn)
                elo['traceback'][i] = 'passing'
                scores.append(score)
            except Exception as e:
                elo['traceback'][i] = traceback.format_exc()

        if len(scores)==2:
            small = tol * (abs(scores[0]) + abs(scores[1]))  # Ties
            points = 1 if scores[0] < scores[1] - small else 0 if scores[1] < scores[0] - small else 0.5
            elo1, elo2 = elo['rating'][i1], elo['rating'][i2]
            min_games = min(elo['count'][i1],elo['count'][i2])
            K = 16 if min_games > 25 else 25
            elo['rating'][i1], elo['rating'][i2] = elo_update(elo1, elo2, points,K)
            elo['count'][i1] += 1
            elo['count'][i2] += 1

    return elo


def elo_expected(d:float,f:float=400)->float:
    """ Expected points scored in a match
    :param d:   Difference in rating
    :param f:   "F"-Factor
    :return:    Expected points
    """
    return 1. / (1 + 10 ** (d / f))


def elo_update(white_elo, black_elo, points, K=25):
    """
    :param white_elo:
    :param black_elo:
    :param points:     1 if White wins, 0 if Black wins
    :param K:
    :return:
    """
    d = black_elo - white_elo
    expected_points = elo_expected(d=d, f=400)
    w = points - expected_points
    white_new_elo = white_elo + K * w
    black_new_elo = black_elo - K * w
    return white_new_elo, black_new_elo

