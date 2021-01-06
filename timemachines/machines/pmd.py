import pmdarima as pm
from timemachines.conventions import to_int_log_space
from timemachines.demonstrating import prior_plot_exogenous
import math


# TODO: https://alkaline-ml.com/pmdarima/auto_examples/example_pipeline.html
# TODO: Include exogenous


def pmd_auto(y, s, k, a, t, e, r):
    """ One way to use pmdarima auto_arima

            - Contemporaneous y[1:] variables are used as exogenous 'X' in pmdarima
            - This only works for k=1

        :returns: x, s', w

    """

    if s is None:
        s = dict()

        # Interpret hyper-parameters - r is only used once at initiation of model
        p_bounds, q_bounds, fit_frequency_bounds = (1, 10), (1, 10), (1, 500)
        s['burnin'] = 50
        s['alpha'] = 0.25          # Defines confidence interval
        s['buffer_len'] = 5000
        s['max_p'], s['max_q'], s['fit_frequency'] = to_int_log_space(r,
                                                                      bounds=[p_bounds, q_bounds, fit_frequency_bounds])

        # Initialize history buffer
        try:
            s['dim'] = len(y)
        except TypeError:
            s['dim'] = 1
        s['buffer'] = list()        # Target
        if s['dim'] > 1:
            s['exogenous'] = list()  # Exogenous
            assert k==1, 'Currently, pmd auto_arima only supports use of exogenous variables when predicting one step ahead'
        s['model']   = None
        s['advance'] = list()  # Variables known in advance
        s['since_last_fit'] = 0

    if y is not None:
        # Process observation and return prediction
        assert isinstance(y, float) or len(y) == s['dim'], ' Cannot change dimension of input in flight '

        if s['dim'] == 1:
            y0 = y
            exog = None
        else:
            y0 = y[0]
            exog = [y[1:]]

        s['buffer'].append(y0)
        if exog is not None:
            s['exogenous'].append(exog[0])
        if a is not None:
            s['advance'].append(a)

        # Do we need to fit?
        s['since_last_fit'] += 1
        first_time = len(s['buffer']) == s['burnin']
        stale = (len(s['buffer']) >= s['burnin']) and (s['since_last_fit'] >= s['fit_frequency'])
        if first_time or stale:
            none_, s, _ = pmd_auto(y=None, s=s, k=k, a=a, t=t, e=e, r=r)  # Fit the model
            assert none_ is None
        elif s['model'] is not None:
            s['model'].update([y0], X=exog)

        if s.get('model') is None:
            return y0, s, None
        else:
            x, ntvl = s['model'].predict(n_periods=k, X=exog, return_conf_int=True,alpha=s['alpha'])
            w = ntvl[k-1][1]-ntvl[k-1][0]
            return x, s, w

    if y is None:
        # Fitting the model
        X = s.get('exogenous') or None
        s['model'] = pm.auto_arima(y=s['buffer'], X=X, start_p=1, start_q=1, start_P=1, start_Q=1,
                                   max_p=s['max_p'], max_q=s['max_q'], max_P=5, max_Q=5, seasonal=True,
                                   stepwise=True, suppress_warnings=True, D=10, max_D=10,
                                   error_action='ignore')
        s['since_last_fit'] = 0
        #print(s['model'].params())
        return None, s, None  # Acknowledge that a fit was requested by returning x=None, w=None


if __name__ == '__main__':
    err = prior_plot_exogenous(f=pmd_auto, k=1, n=500, r=0.05)
    pass
