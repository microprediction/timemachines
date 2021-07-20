
NPROPHET_META={'n_warm':20,  # Default minimum number of data points before fitting
              'freq':'H',    # Sampling frequency used if none is supplied, or t not supplied
              'n_max':1000}   # Maximum length of training data


NPROPHET_MODEL = dict(n_lags=2,
    changepoints_range=0.95,
    n_changepoints=30,
    weekly_seasonality=False,
    batch_size=64,
    epochs=10,
    learning_rate=1.0)