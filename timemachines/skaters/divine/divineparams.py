
MIN_N_WARM = 101   # Library will fail if history is too small

DIVINE_MODEL = dict(confidence_interval = 34.134*2,   # So it is +/- one standard deviation
                    seasonal_periods = [7, 14., 28., 30., 90., 120., 182., 365.],
                     optimise_trend_season_features=True,
                    trend_order=[0, 1]
                    )
