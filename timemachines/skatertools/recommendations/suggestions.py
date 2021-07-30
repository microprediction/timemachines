K_LEADERBOARD_TYPES = ['univariate','residual']
from getjson import getjson
from timemachines.skaters.localskaters import local_skater_from_name


# Utilities for grabbing suggested models from the Elo ratings


FIBONACCI = [ 1, 2, 3, 5, 8, 13, 21, 34]  # k-s used in the ratings


def closest(lst, k):
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - k))]


def closest_fib(k):
    return closest(FIBONACCI,k)


def ratings_url(k:int, category='univariate'):
    url_template = 'https://raw.githubusercontent.com/microprediction/timeseries-elo-ratings/main/ratings/CATEGORY-k_KKK.json'
    kkk = str(k).zfill(3)
    return url_template.replace('KKK',kkk).replace('CATEGORY', category)


def get_ratings(k:int, category='univariate'):
    url = ratings_url(k=k,category=category)
    return getjson(url=url)


def top_rated(k:int, category='univariate', max_seconds=10, min_count=10, require_passing:bool=False,
              ignore_elo=True):
    """
    :param k:                number of steps ahead to forecast
    :param category:        'univariate' or 'residual'
    :param max_seconds:      for about 50 separate forecasts
    :param min_count:        minimum number of games played to establish the Elo rating
    :param require_passing:  filter out if the tests are failing
    :param ignore_elo        filter out skaters with elo in name, to avoid infinite recursion
    :return:    [ (rating, name, package_url) ]
    """
    k_closest = closest_fib(k)
    rd = get_ratings(k=k_closest,category=category)
    rd_zip = zip( rd['name'], rd['count'], rd['rating'],rd['traceback'],rd['seconds'], rd['pypi'] )
    return sorted([(rtng,nm,pypi) for nm,cnt,rtng,trcbck,scnds,pypi in rd_zip if
                   ((scnds>=0) and (scnds<=max_seconds)) and
                   (trcbck=='passing' or (not require_passing)) and
                   (cnt>=min_count) and
                   ((not 'elo' in nm) or not ignore_elo)
                   ], reverse=True )


def top_rated_names(k:int, n=5, category='univariate', max_seconds=10, min_count=10, require_passing:bool=False, ignore_elo=False)->[str]:
    rcm = top_rated(k=k, category=category, max_seconds=max_seconds, min_count=min_count, require_passing=require_passing,ignore_elo=ignore_elo)
    return [ r[1] for r in rcm[:n]]


def top_rated_models(k:int, n=5, category='univariate', max_seconds=10, min_count=10, require_passing:bool=False, ignore_elo=False):
    """
       Try to instantiate top skaters, moving down the list until we have at least n

       :returns  [ skaters ]
    """
    rcm = top_rated(k=k, category=category, max_seconds=max_seconds, min_count=min_count, require_passing=require_passing, ignore_elo=ignore_elo)
    loaded = list()

    for r in rcm:
        try:
            skater = local_skater_from_name(r[1])
            if skater is not None:
                loaded.append(skater)
                if len(loaded)==n:
                    break
        except ImportError:
            print('Cannot instantiate recommendation ' +r[1]+'. For install instructions see the package '+r[2])

    return loaded


if __name__=='__main__':
    my_skaters = top_rated_models(k=3, n=5)
    print(my_skaters)



