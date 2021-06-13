from timemachines.skaters.dlm.dlminclusion import using_dlm, dlm, trend, seasonality, autoReg
from timemachines.skatertools.utilities.conventions import dimension, to_int_log_space, split_exogenous, Y_TYPE, R_TYPE
from timemachines.skatertools.visualization.priorplot import prior_plot_exogenous
import math
import numpy as np

if using_dlm:

    from timemachines.skaters.dlm.dlmhacks import fixedAutoReg

    ###################################################
    #                                                 #
    #      Univariate skater based on pydlm           #
    #                                                 #
    ###################################################


    def dlm_univariate_a(y, s: dict, k: int, a=None, t=None, e=None):
        ra = 0.25
        return dlm_univariate_r3(y=y, s=s, k=k, r=ra)


    def dlm_univariate_b(y, s: dict, k: int, a=None, t=None, e=None):
        ra = 0.75
        return dlm_univariate_r3(y=y, s=s, k=k, r=ra)


    def dlm_univariate_r3(y, s: dict, k: int, a=None, t=None, e=None, r=None):
        """ Univariate filter

                - Uses the discounting method of H/W so, doesn't need to be fit as often
                - Discount factors are periodically tuned
                - The hyper-parameter controls 'auto_degree', 'trend_degree',  'period'

            :returns: x, x_std, s
        """
        assert r is not None, 'Requires hyper-parameter (interpreted in dimension 3) '
        if not s:
            s = dict()
            s = dlm_set_univariate_params(s=s, r=r)
            s['dim'] = dimension(y)
            s['n_obs'] = 0
            s['model'] = dlm([], printInfo=False) + trend(s['trend_degree'], s['discount']) + seasonality(s['period'],
                                                                                                          s['discount'])
            s['model'] = s['model'] + fixedAutoReg(degree=s['auto_degree'], name='ar', w=1.0)

        if y is not None:
            s['n_obs'] += 1
            assert isinstance(y, float) or len(y) == s['dim'], ' Cannot change dimension of input in flight '
            y0, exog = split_exogenous(y=y)
            y0_passed_in = None if np.isnan(y0) else y0  # pydlm uses None for missing values
            s['model'].append([y0_passed_in])
            num_obs = len(s['model'].data) if s.get('model') else 0
            if num_obs % s['n_fit'] == s['n_fit'] - 1:
                # Perform periodic tuning of discount factors
                _, _, s = dlm_univariate_r3(y=None, s=s, k=k, a=a, t=t, e=1000, r=r)
            s['model'].fitForwardFilter()
            return _dlm_prediction_helper(s=s, k=k, y=y)

        if y is None and e > 60:
            s['model'].tune()  # Tunes discount factors
            s['model'].fit()
            return None, None, s


    def _dlm_prediction_helper(s, k:int, y:Y_TYPE):
        """ Calls down to predictN, if we are passed the warm-up stage """
        num_obs = len(s['model'].data) if s.get('model') else 0
        if num_obs < s['n_warm']:
            y0, _ = split_exogenous(y)
            return [y0]*k, [abs(y0)]*k, s
        else:
            x_mean, x_var = s['model'].predictN(N=k)
            x = list(x_mean)
            x_std = [ math.sqrt(v) for v in x_var]
            return x, x_std, s



    def dlm_set_univariate_params(s, r:R_TYPE):
        period_choices = [3,5,7,10,12,16,24,32]
        s['auto_degree'], s['trend_degree'], period_choice = to_int_log_space(r,bounds=[
            (0.5,3.2),    # auto-reg degree. 1 means linear trend, 2 quadratic
            (0.1,3.4),    # seasonality degree
            (0,len(period_choices)) # seasonality period
        ])
        s['discount'] = 1 - ( r % 0.05 ) # Shouldn't matter too much as these get tuned eventually
        s['period'] = period_choices[period_choice]
        s['n_warm'] = 100
        s['n_fit']  = 500  # How often to tune discounts subsequently
        return s


if __name__ == '__main__':
    assert using_dlm, 'pip install pydlm'
    import matplotlib.pyplot as plt
    err = prior_plot_exogenous(f=dlm_univariate_r3, k=2, n=500, r=np.random.rand())
    plt.figure()
    print('done')
    pass
