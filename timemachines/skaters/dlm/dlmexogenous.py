from timemachines.skaters.dlm.dlminclusion import using_dlm, trend, seasonality, dynamic
if using_dlm:
    from timemachines.skatertools.utilities.conventions import dimension, to_int_log_space, split_exogenous, Y_TYPE, R_TYPE
    from timemachines.skatertools.visualization.priorplot import prior_plot_exogenous
    from timemachines.skaters.dlm.dlmhacks import quietDlm, fixedAutoReg
    from timemachines.skatertools.utilities.conventions import wrap
    import math
    import numpy as np


    def dlm_exogenous_a(y, s: dict, k: int, a=None, t=None, e=None):
        ra = 0.25
        return dlm_exogenous_r3(y=y, s=s, k=k,  a=a, t=t, e=e, r=ra)


    def dlm_exogenous_b(y, s: dict, k: int, a=None, t=None, e=None):
        rb = 0.75
        return dlm_exogenous_r3(y=y, s=s, k=k, a=a, t=t, e=e, r=rb)


    def dlm_exogenous_r3(y, s, k, a, t, e, r):
        """ One way to use dlm
            :returns: x, s', w
        """
        if not s:
            s = dict()
            s['dim'] = dimension(y)
            s = dlm_set_exog_hyperparams(s=s, r=r)
            y0, exog = split_exogenous(y=y)
            s['n_obs'] = 0
            s['model'] = quietDlm([], printInfo=False) + trend(s['trend_degree'], s['discount']) + seasonality(s['period'], s['discount'])
            s['model'] = s['model'] + fixedAutoReg(degree=s['auto_degree'], name='ar', w=1.0)
            if exog:
                exog_wrapped = [[None if np.isnan(ex0) else ex0 for ex0 in exog]]
                s['model'] = s['model'] + dynamic(features=exog_wrapped, discount=0.99, name='exog') # Set's first exog

        if y is not None:
            y = wrap(y)
            assert dimension(y)==s['dim'],'Cannot change dimension of data sent'
            s['n_obs'] += 1
            y0, exog = split_exogenous(y=y)
            y0_passed_in = None if np.isnan(y0) else y0  # pydlm uses None for missing values
            s['model'].append([y0_passed_in])
            if exog:
                exog_wrapped = [[ None if np.isnan(ex0) else ex0 for ex0 in exog ]]
                if s['n_obs']>1:
                    s['model'].append(data=exog_wrapped, component='exog') # Don't get first exog twice
            num_obs = len(s['model'].data) if s.get('model') else 0
            if num_obs % s['n_fit'] == s['n_fit']-1:
                _, _, s = dlm_exogenous_r3(y=None, s=s, k=k, a=a, t=t, e=10, r=r)
            s['model'].fitForwardFilter()
            return _dlm_exog_prediction_helper(s=s, k=k, y=y)

        if y is None:
            if dimension(y)==1:
                s['model'].tune(maxit=20)
                # Don't tune if exogenous ... haven't got this to work
            s['model'].fit()
            return None, None, s



    def _dlm_exog_prediction_helper(s, k:int, y:Y_TYPE):
        num_obs = len(s['model'].data) if s.get('model') else 0
        if num_obs < s['n_burn']:
            y0, exog = split_exogenous(y)
            return [y0]*k, [abs(y0)]*k, s
        else:
            assert k==1,'only k==1 for now' # TODO: Fix to allow for k>1
            y0, exog = split_exogenous(y)
            if exog:
                exog_passed_in = [None if np.isnan(ex0) else ex0 for ex0 in exog]
                x_mean, x_var = s['model'].predict(featureDict={'exog':exog_passed_in[0]})
            else:
                x_mean, x_var = s['model'].predict()
            x = [x_mean[0, 0]]
            x_std = [ math.sqrt( x_var[0,0] ) ]

            return x, x_std, s


    def dlm_set_exog_hyperparams(s, r:R_TYPE):
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



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    err = prior_plot_exogenous(f=dlm_exogenous_r3, k=1, n=1000, r=np.random.rand())
    plt.figure()
    print('done')
    pass
