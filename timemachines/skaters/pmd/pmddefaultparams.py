
PMD_CRITERIA = ['aic', 'aicc', 'bic', 'hqic']


def pmd_params(method=None):
    """ Default hyper-params """
    # https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html
    s = dict(start_p=1, start_q=1, start_P=1, start_Q=1,
                       max_p=10, max_q=10, max_P=24, max_Q=24, seasonal=False,
                       stepwise=True, suppress_warnings=True,
                       D=None, max_D=3,
                       error_action='ignore',
                       information_criterion=PMD_CRITERIA[0])
    if method == '0':
        # Add more cases here
        pass
    return s