from timemachines.skaters.proph.prophiskaterfactory import using_prophet
if using_prophet:
    from timemachines.skaters.proph.prophskaterfactory import fbprophet_skater_factory, fbprophet_hyperparam_skater_factory
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE, wrap
    import numpy as np

    # A collection of skaters powered by fbprophet
    # You'll want to read this review: https://www.microprediction.com/blog/prophet


    def fbprophet_exogenous(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        """ Uses known-in-advance and also y[1:] brought forward """
        return fbprophet_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e)


    def fbprophet_recursive(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        """ Same as exogenous, but uses prophet to predict y[1:]  """
        return fbprophet_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, recursive=True)


    def fbprophet_known(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        """ Uses known-in-advance but not y[1:] """
        y0 = [wrap(y)[0]]
        return fbprophet_skater_factory(y=y0, s=s, k=k, a=a, t=t, e=e)


    def fbprophet_univariate(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        """ Simple univariate prediction using only y[0], and not 'a' or y[1:] """
        y0 = [wrap(y)[0]]
        return fbprophet_skater_factory(y=y0, s=s, k=k, a=None, t=t, e=e)



    def fbprophet_cautious(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        """ Similar to fbexogenous, but no crazy nonsense """
        if not s.get('s'):
            s['s']={}       # prophet's state
            s['y']=list()   # maintain last five values
        y0 = wrap(y)[0]
        s['y'].append(y0)
        if len(s['y'])>5:
            s['y'].pop(0)
        import math
        x_upper = [ np.max(s['y'])+math.sqrt(j+1)*np.std(s['y']) for j in range(k) ]
        x_lower = [ np.min(s['y'])-math.sqrt(j+1)*np.std(s['y']) for j in range(k) ]
        x, x_std, s['s'] = fbprophet_univariate(y=y,s=s['s'],k=k,a=a,t=t,e=e)
        x_careful = np.minimum(np.array(x),np.array(x_upper))
        x_careful = np.maximum(x_careful, np.array(x_lower))
        return list(x_careful), x_std, s


    PROPHET_SKATERS_SINGULAR = [fbprophet_exogenous, fbprophet_known, fbprophet_univariate, fbprophet_recursive,
                                fbprophet_cautious]


    # (1) Skaters with author-suggested two-dimensional hyper-parameter spaces

    def fbprophet_exogenous_r2(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r:R_TYPE=None):
        """ A skater using exogenous variables, with hyper-param tuning as recommended by authors """
        assert r is not None
        param_names = ['changepoint_prior_scale', 'seasonality_prior_scale']
        return fbprophet_hyperparam_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, param_names=param_names)


    def fbprophet_recursive_r2(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r:R_TYPE=None):
        """ Same as exogenous, but uses prophet to predict y[1:]  """
        assert r is not None
        param_names = ['changepoint_prior_scale', 'seasonality_prior_scale']
        return fbprophet_hyperparam_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, param_names=param_names, recursive=True)


    def fbprophet_known_r2(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r:R_TYPE=None):
        """ Uses known-in-advance but not y[1:] """
        assert r is not None
        y0 = [wrap(y)[0]]
        param_names = ['changepoint_prior_scale', 'seasonality_prior_scale']
        return fbprophet_hyperparam_skater_factory(y=y0, s=s, k=k, a=a, t=t, e=e, r=r, param_names=param_names, recursive=False)


    def fbprophet_univariate_r2(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None,r:R_TYPE=None):
        """ Simple univariate prediction using only y[0], and not 'a' or y[1:] """
        assert r is not None
        y0 = [wrap(y)[0]]
        param_names = ['changepoint_prior_scale', 'seasonality_prior_scale']
        return fbprophet_hyperparam_skater_factory(y=y0, s=s, k=k, a=None, t=t, e=e, r=r, param_names=param_names, recursive=False)


    PROPHET_R2_SKATERS = [ fbprophet_exogenous_r2, fbprophet_known_r2, fbprophet_univariate_r2, fbprophet_recursive_r2 ]
else:
    PROPHET_SKATERS_SINGULAR = []
    PROPHET_R2_SKATERS = []



if __name__ == '__main__':
    from timemachines.skatertools.data import hospital_with_exog
    from timemachines.skatertools.visualization.priorplot import prior_plot
    import matplotlib.pyplot as plt
    k = 1
    y, a = hospital_with_exog(k=k, n=450, offset=True)
    f = fbprophet_exogenous
    err2 = prior_plot(f=f, k=k, y=y, n=450, n_plot=50)
    print(err2)
    plt.show()
    pass


