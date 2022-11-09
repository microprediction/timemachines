# Compute performance rating for a single algorithm

from timemachines.skatertools.comparison.skaterelo import _init_elo
from timemachines.skatertools.recommendations.suggestions import get_ratings, show_ratings
from pprint import pprint
from timemachines.skatertools.comparison.skaterelo import skater_elo_multi_update, DEFAULT_INITIAL_ELO

DEFAULT_BENCHMARK_SKATER_NAMES = ['sluggish_moving_average',
                                  'thinking_fast_and_slow',
                                   'aggressive_ema_ensemble',
                                  'tsa_balanced_theta_ensemble',
                                  'thinking_fast_and_fast',
                                  'empirical_last_value',
                                  'rapidly_moving_average']


def performance_rating(skater_names:[str],k=1, category='univariate', initial_elo=DEFAULT_INITIAL_ELO,
                       benchmark_skater_names=None, update_frequency=10):
    # Play in a small pool that includes some rated skaters
    from timemachines.skaters.allskaters import skater_from_name
    if benchmark_skater_names is None:
        benchmark_skater_names = DEFAULT_BENCHMARK_SKATER_NAMES
    all_elo = get_ratings(k=k, category=category)
    ndx = [ i for i,nm in enumerate(all_elo['name']) if nm in benchmark_skater_names ]
    elo = dict([ (ky, [ v[i] for i in ndx] ) for ky,v in all_elo.items() if ky not in ['evaluator'] ])

    benchmark_skater_attempted = [ skater_from_name(nm) for nm in benchmark_skater_names ]
    benchmark_skater_population = [ f for f in benchmark_skater_attempted if not f is None ]
    benchmark_skater_failed = [ nm for nm, a in zip(benchmark_skater_names,benchmark_skater_attempted) if a is None]
    if len(benchmark_skater_failed):
        print('The following benchmark skaters failed to load. You might need to install some requisite package')
        pprint(benchmark_skater_failed)

    new_skater_attempted = [ skater_from_name(nm) for nm in skater_names ]
    new_skater_population = [ f for f in new_skater_attempted if not f is None ]
    new_skater_failed = [ nm for nm, a in zip(skater_names,new_skater_attempted) if a is None]
    if len(new_skater_failed):
        print('The following new skaters to load. You might need to install some requisite package')
        pprint(new_skater_failed)

    new_elo = _init_elo(elo={}, skater_population=new_skater_population, initial_elo=initial_elo)
    for ky in ['name','count','rating','traceback','active','pypi','seconds']:
        elo[ky] = elo[ky] + new_elo[ky]

    while True:
        import random
        if random.choice(range(update_frequency))==0:
            rat = show_ratings(elo)
            some_rat = [ r for r in rat if r[1] in benchmark_skater_names+skater_names]
            print('')
            print('---- Ratings -----')
            pprint(some_rat)
        elo = skater_elo_multi_update(elo=elo, k=k, evaluator=None, n_burn=400, tol=0.01, initial_elo=1600,
                                data_source=None, skater_population=benchmark_skater_population, always_skaters=None, verbose=False)


if __name__=='__main__':
    from timemachines.skaters.simple.thinking import WIGGLY_THINKING_FAST_AND_SLOW_SKATERS
    skater_names = [f.__name__ for f in WIGGLY_THINKING_FAST_AND_SLOW_SKATERS]
    performance_rating(skater_names=skater_names)
