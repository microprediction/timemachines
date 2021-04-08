
def nproph_params(method=None):
    """ Default hyper-params """
    # https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html
    s = dict(
        nlags=4,
        changepoints_range=0.95,
        n_changepoints=30,
        weekly_seasonality=False,
        batch_size=64,
        epochs=10,
        learning_rate=1.0
    )
    if method == '0':
        # Add more cases here
        pass
    return s