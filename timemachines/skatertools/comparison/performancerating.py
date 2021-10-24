# Compute performance rating for a single algorithm

# Not done

def performance_rating(skater_names:[str]):
    # Play in a small pool

    benchmark_skater_names = ['merlion_arima','tsa_p2_d0_q0','sluggish_moving_average','thinking_slow_and_fast',
                              'aggressive_ema_ensemble','tsa_balanced_theta_ensemble','thinking_fast_and_fast','empirical_last_value',
                              'rapidly_moving_average']
    from timemachines.skaters.allskaters import skater_from_name
    skater_population = [ skater_from_name(nm) for nm in benchmark_skater_names+skater_names ]
    from timemachines.skatertools.comparison.skaterelo import skater_elo_multi_update


