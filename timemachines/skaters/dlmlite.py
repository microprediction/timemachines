from pydlm import dlm, trend, seasonality
from timemachines.conventions import dimension, to_log_space, separate_observations, update_buffers, \
    Y_TYPE, A_TYPE, R_TYPE
from timemachines.plotting import prior_plot_exogenous
import math
import numpy as np
import matplotlib.pyplot as plt

# pydlm-lite skaters
# See https://github.com/wwrechard/pydlm-lite for explanation of the refactoring of pydlm
# The new version might be better for performant skaters, though it is fluid.


def dlm_seasonal_hyperparams(s, r:R_TYPE):
    # One way to interpret hyper-params r for pmd_arima models
    # e.g. dlm([]) + trend(degree=2, 0.98) + + seasonality(period = 7, 0.98)
    degree_bounds = (0.1,4.1)
    initial_discount = (0.90,0.97)  # Tuned subsequently so doesn't matter that much
    seasonality_choices = [0,7,24,3*24,4*24]
    seasonality_choice_bounds = [0.2,4.4]
    bounds=[seasonality_choice_bounds, degree_bounds, initial_discount ]
    _period_choice, _degree, s['discount'] = to_log_space(r, bounds=bounds)
    period_choice = int(_period_choice)
    assert 0<= period_choice <= 3
    s['degree'], s['period'] = int(_degree), seasonality_choices[period_choice]
    s['n_burn'] = max(50,2*s['period'])
    s['n_fit'] = 100  # How often to tune discount
    return s


def dlm_seasonal_or_last_value(s, k:int, y:Y_TYPE):
    # Predict using model, or fall back to last value
    num_obs = len(s['model'].data) if s.get('model') else 0
    if num_obs < s['n_burn']:
        y0, _ = separate_observations(y,s['dim'])
        return y0, s, None
    else:
        assert k==1,'only k==1 for now' # TODO: Fix
        x_mean, x_var = s['model'].predict()
        x = x_mean[0, 0]
        v = x_var[0,0]
        try:
            w = math.sqrt(v)
        except:
            print('Warning - negative var returned '+str(v))
            w = math.sqrt(abs(v))
        return x, s, w


def dlm_seasonal(y, s, k, a, t, e, r):
    """ One way to use dlm
        :returns: x, s', w
    """
    if s is None:
        s = dict()
        s = dlm_seasonal_hyperparams(s=s, r=r)
        s['dim'] = dimension(y)
        if s['period']>0:
            s['model'] = dlm([]) + trend(s['degree'], s['discount']) + seasonality(s['period'], s['discount'])
        else:
            s['model'] = dlm([]) + trend(s['degree'], s['discount'])

    if y is not None:
        assert isinstance(y, float) or len(y) == s['dim'], ' Cannot change dimension of input in flight '
        y0, exog = separate_observations(y=y,dim=s['dim'])
        y0_passed_in = None if np.isnan(y0) else None  # pydlm uses None for missing values
        s['model'].append([y0])
        num_obs = len(s['model'].data) if s.get('model') else 0
        if num_obs % s['n_fit'] == s['n_fit']-1:
            _, s, _ = dlm_seasonal(y=None,s=s,k=k,a=a,t=t,e=10,r=r) # Fit
        s['model'].fitForwardFilter()
        s['model'].fitBackwardSmoother()
        return dlm_seasonal_or_last_value(s=s, k=k, y=y)

    if y is None:
        s['model'].tune() # Tunes discount factors
        s['model'].fit()
        return None, s, None


if __name__ == '__main__':
    err = prior_plot_exogenous(f=dlm_seasonal, k=1, n=200, r=0.05)
    plt.figure()
    print('done')
    pass
