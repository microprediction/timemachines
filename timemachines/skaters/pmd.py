import pmdarima as pm
from timemachines.conventions import to_int_log_space, initialize_buffers, separate_observations, update_buffers, \
    Y_TYPE, A_TYPE, R_TYPE
from timemachines.plotting import prior_plot_exogenous


# pmd-arima skaters


def pmd_hyperparams(s, r:R_TYPE):
    # One way to interpret hyper-params r for pmd_arima models
    p_bounds, q_bounds, n_fit_bounds = (1, 10), (1, 10), (1, 500)
    s['n_burn'] = 50
    s['alpha'] = 0.25  # Defines confidence interval
    s['buffer_len'] = 5000
    s['max_p'], s['max_q'], s['n_fit'] = to_int_log_space(r, bounds=[p_bounds, q_bounds, n_fit_bounds])
    return s


def pmd_or_last_value(s, k:int, exog:[float], y0:float):
    # Predict k steps ahead, or fall back to last value
    if s.get('model') is None:
        return y0, s, None
    else:
        x, ntvl = s['model'].predict(n_periods=k, X=exog, return_conf_int=True, alpha=s['alpha'])
        w = ntvl[k - 1][1] - ntvl[k - 1][0]
        return x, s, w


def pmd_auto(y, s, k, a, t, e, r):
    """ One way to use pmdarima auto_arima

            - Contemporaneous y[1:] variables are used as exogenous 'X' in pmdarima
            - This only works for k=1

        :returns: x, s', w
    """
    if s is None:
        s = dict()
        s = pmd_hyperparams(s=s, r=r)
        s = initialize_buffers(s=s,y=y)

    if y is not None:
        assert isinstance(y, float) or len(y) == s['dim'], ' Cannot change dimension of input in flight '
        y0, exog = separate_observations(y=y, dim=s['dim'])
        s = update_buffers(s=s,a=a,exog=exog,y0=y0)
        s['staleness'] += 1
        stale = (len(s['buffer']) >= s['n_burn']) and (s['staleness'] >= s['n_fit'])
        first_time = len(s['buffer']) == s['n_burn']
        if first_time or stale:
            none_, s, _ = pmd_auto(y=None, s=s, k=k, a=a, t=t, e=e, r=r)  # Fit the model
            assert none_ is None
        elif s['model'] is not None:
            s['model'].update([y0], X=exog)
        return pmd_or_last_value(s=s, k=k, exog=exog, y0=y0)

    if y is None:
        # Fitting the model
        X = s.get('exogenous') or None
        s['model'] = pm.auto_arima(y=s['buffer'], X=X, start_p=1, start_q=1, start_P=1, start_Q=1,
                                   max_p=s['max_p'], max_q=s['max_q'], max_P=5, max_Q=5, seasonal=True,
                                   stepwise=True, suppress_warnings=True, D=10, max_D=10,
                                   error_action='ignore')
        s['staleness'] = 0
        return None, s, None


if __name__ == '__main__':
    err = prior_plot_exogenous(f=pmd_auto, k=1, n=500, r=0.05)
    pass
