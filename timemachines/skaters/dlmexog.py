from pydlm import dlm, trend, seasonality, autoReg, dynamic
from timemachines.conventions import dimension, to_int_log_space, separate_observations, update_buffers, \
    Y_TYPE, A_TYPE, R_TYPE
from timemachines.plotting import prior_plot_exogenous
import math
import numpy as np
import matplotlib.pyplot as plt


class fixedAutoReg(autoReg):

    def __init__(self,*arg,**kwargs):
        super().__init__(*arg,**kwargs)

    def appendNewData(self, data):
        """ AutoReg append new data automatically with the main time series. Nothing
        needs to be done here.
        """
        return


def dlm_exog_hyperparams(s, r:R_TYPE):
    # Univariate model with autoregressive components
    # This uses the discounting method of H/W so doesn't need to be fit as often
    period_choices = [3,5,7,10,12,16,24,32]
    s['auto_degree'], s['trend_degree'], period_choice = to_int_log_space(r,bounds=[
        (0.5,3.2),    # auto-reg degree. 1 means linear trend, 2 quadratic
        (0.1,3.4),    # seasonality degree
        (0,len(period_choices)) # seasonality period
    ])
    s['discount'] = 1 - ( r % 0.05 ) # Shouldn't matter too much as these get tuned eventually
    s['period'] = period_choices[period_choice]
    s['n_burn'] = 100
    s['n_fit']  = 500  # How often to tune discounts subsequently
    return s


def dlm_exog_or_last_value(s, k:int, y:Y_TYPE):
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


def dlm_exog(y, s, k, a, t, e, r):
    """ One way to use dlm
        :returns: x, s', w
    """
    if s is None:
        s = dict()
        s['dim'] = dimension(y)
        s = dlm_exog_hyperparams(s=s, r=r)
        y0, exog = separate_observations(y=y, dim=s['dim'])
        y0_passed_in = None if np.isnan(y0) else y0  # pydlm uses None for missing values
        exog_passed_in = [None if np.isnan(ex0) else ex0 for ex0 in exog]
        s['n_obs'] = 0
        s['model'] = dlm([],printInfo=False) + trend(s['trend_degree'], s['discount']) + seasonality(s['period'], s['discount'])
        s['model'] = s['model'] + fixedAutoReg(degree=s['auto_degree'], name='ar', w=1.0)
        s['model'] = s['model'] + dynamic(features=exog_passed_in, discount=0.99, name='exog') # Set's first exog

    if y is not None:
        s['n_obs'] += 1
        assert isinstance(y, float) or len(y) == s['dim'], ' Cannot change dimension of input in flight '
        y0, exog = separate_observations(y=y,dim=s['dim'])
        y0_passed_in = None if np.isnan(y0) else y0  # pydlm uses None for missing values
        exog_passed_in = [ None if np.isnan(ex0) else ex0 for ex0 in exog ]
        s['model'].append([y0_passed_in])
        if s['n_obs']>1:
            s['model'].append(data=exog_passed_in, component='exog') # Don't get first exog twice
        num_obs = len(s['model'].data) if s.get('model') else 0
        if num_obs % s['n_fit'] == s['n_fit']-1:
            _, s, _ = dlm_exog(y=None,s=s,k=k,a=a,t=t,e=10,r=r)
        s['model'].fitForwardFilter()
        return dlm_exog_or_last_value(s=s, k=k, y=y)

    if y is None:
        s['model'].tune() # Tunes discount factors
        s['model'].fit()
        return None, s, None


if __name__ == '__main__':
    err = prior_plot_exogenous(f=dlm_exog, k=1, n=1000, r=np.random.rand())
    plt.figure()
    print('done')
    pass
