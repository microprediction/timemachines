import pmdarima as pm
from timemachines.conventions import to_space, initialize_buffers, separate_observations, update_buffers, \
    Y_TYPE, A_TYPE, R_TYPE
from timemachines.plotting import prior_plot_exogenous


# pmd-arima skaters


def pmd_hyperparams(s, r:R_TYPE):
    # One way to interpret hyper-params r for pmd_arima models

    m_float, criterion, stepwise = to_space(r, bounds=[ (0.5,12+0.5),(-0.5,3.5), (0,1)])
    CRITERIA = ['aic','aicc','bic','hqic']
    s['information_criterion'] = CRITERIA[int(criterion)]
    s['seasonal'] = True
    s['m'] = int(m_float)
    s['stepwise'] = stepwise<0.5
    s['n_burn'] = 100     # Initial burn in period before first fit
    s['n_fit'] = 500      # Time between fits
    s['alpha'] = 0.25     # Defines confidence interval
    s['buffer_len'] = 5000
    return s


def pmd_or_last_value(s, k:int, exog:[float], y0:float):
    # Predict k steps ahead, or fall back to last value
    if s.get('model') is None:
        return y0, s, None
    else:
        x, ntvl = s['model'].predict(n_periods=k, X=exog, return_conf_int=True, alpha=s['alpha'])
        w = ntvl[k - 1][1] - ntvl[k - 1][0]
        return x[0], s, w


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
                                   max_p=10, max_q=10, max_P=24, max_Q=24, seasonal=s['seasonal'],
                                   stepwise=s['stepwise'], suppress_warnings=True, D=1, max_D=3, m=s['m'],
                                   error_action='ignore', information_criterion=s['information_criterion'])
        print(s['model'])
        print(s['model'].params())
        s['staleness'] = 0
        return None, s, None


if __name__ == '__main__':
    prior_plot_exogenous(f=pmd_auto, k=1, n=1000, r=0.95)

    if False:
        from timemachines.evaluation import evaluate_mean_absolute_error
        from timemachines.data.real import hospital
        from timemachines.skating import prior
        ys = hospital()[:550]
        xs1 = prior(f=pmd_auto,ys=ys, k=1, r=0.95)
        xs2 = prior(f=pmd_auto,ys=ys, k=1, r=0.05)
        pass
        err1 = evaluate_mean_absolute_error(f=pmd_auto,k=1, ys=ys, r=0.95, n_burn=250)
        print('----------------')
        err2 = evaluate_mean_absolute_error(f=pmd_auto, k=1, ys=ys, r=0.05, n_burn=250)
        print((err1,err2))
