from timemachines.skaters.pmd.pmdinclusion import using_pmd, pm
if using_pmd:
    from timemachines.skaters.pmd.pmdskaterfactory import pmd_skater_factory
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE,T_TYPE,E_TYPE, wrap
    from timemachines.skaters.simple.movingaverage import empirical_last_value
    from timemachines.skatertools.visualization.priorplot import prior_plot, prior_plot_exogenous


    def pmd_exogenous(y:Y_TYPE, s:dict, k:int=1, a:A_TYPE=None, t:T_TYPE=None, e:E_TYPE=None):
        """ Predict using auto_arima, with both simultaneously observed and known in advance variables
            This skater has no hyper-parameters

            y: Y_TYPE    scalar or list where y[1:] are interpreted as contemporaneously observed exogenous variables
            s:           state
            k:           Number of steps ahead to predict
            a:           (optional) scalar or list of variables known k-steps in advance.
                          (IMPORTANT: If supplying 'a', provide the known variable k steps ahead, not the contemporaneous one !).
            t:           (optional) Time of observation.
            e:           (optional) Maximum computation time (supply e>60 to give hint to do fitting)

            :returns: x [float] , s', scale [float]
        """
        return pmd_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, method='auto')


    def pmd_known(y:Y_TYPE, s:dict, k:int=1, a:A_TYPE=None, t:T_TYPE=None, e:E_TYPE=None):
        """ Uses known-in-advance but not y[1:] """
        y0 = [wrap(y)[0]]
        return pmd_skater_factory(y=y0, s=s, k=k, a=a, t=t, e=e, method='auto')


    def pmd_univariate(y:Y_TYPE, s:dict, k:int=1, a:A_TYPE=None, t:T_TYPE=None, e:E_TYPE=None):
        """ Uses only y[0] and ignores y[1:] and a[:] """
        y0 = [wrap(y)[0]]
        return pmd_skater_factory(y=y0, s=s, k=k, a=None, t=t, e=e, method='auto')


    def pmd_exog_compare(f,k=1):
        from timemachines.skatertools.evaluation.evaluators import evaluate_mean_absolute_error
        from timemachines.skatertools.evaluation.evaluators import hospital_with_exog
        y, a = hospital_with_exog(k=k)
        y0 = [ yi[0] for yi in y ]

        r = 0.1 # Doesn't matter?
        err1 = evaluate_mean_absolute_error(f=f, k=k, y=y0, r=r, n_burn=250)
        err2 = evaluate_mean_absolute_error(f=f, k=k, y=y, r=r, n_burn=250)
        err3 = evaluate_mean_absolute_error(f=f, k=k, y=y, r=r, a=a, n_burn=250)
        errlv = evaluate_mean_absolute_error(f=empirical_last_value, k=k, y=y, r=r, a=a, n_burn=250)


        print('----------------')
        print("Error w/o exogenous   = "+str(err1))
        print("Error w   exogenous   = "+str(err2))
        print("Error w   exo + known = "+str(err3))
        print("Error last val cache  = " + str(errlv))


if __name__ == '__main__':
    assert using_pmd,'pip install pmdarima'
    f = pmd_exogenous
    if True:
        prior_plot_exogenous(f=f, k=1, n=200)
    if True:
        prior_plot(f=f,k=1,n=200)
