
# TODO: Move into ratings package to remove this dependency

def elo_expected(d :float ,f :float =400 )->float:
    """ Expected points scored in a match by White player
    :param d:   Difference in rating (Black minus White)
    :param f:   "F"-Factor
    :return:
    """
    if d/ f > 8:
        return 0.0
    elif d / f < -8:
        return 1.0
    else:
        return 1. / (1 + 10 ** (d / f))


def elo_update(white_elo, black_elo, points, k=25, f=400):
    """
    :param white_elo:
    :param black_elo:
    :param points:     1 if White wins, 0 if Black wins
    :param k:   K-Factor, how much to update
    :param f:   F-Factor
    :return:
    """
    d = black_elo - white_elo
    expected_points = elo_expected(d=d, f=f)
    w = points - expected_points
    white_new_elo = white_elo + k * w
    black_new_elo = black_elo - k * w
    return white_new_elo, black_new_elo
