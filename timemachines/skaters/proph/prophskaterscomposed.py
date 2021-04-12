from timemachines.skaters.proph.prophiskaterfactory import using_prophet
if using_prophet:
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, E_TYPE, T_TYPE
    from timemachines.skatertools.composition.residualshypocratic import quickly_moving_hypocratic_residual_factory
    from timemachines.skatertools.composition.residualcomposition import residual_chaser_factory
    from timemachines.skaters.proph.prophskaterssingular import fbprophet_univariate, fbprophet_exogenous, fbprophet_cautious


    def fbprophet_exogenous_hypocratic(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        """ Chase residuals, somewhat cautiously using, quickly moving average """
        return quickly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=None, f=fbprophet_exogenous)


    def fbprophet_univariate_hypocratic(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        """ Chase residuals, somewhat cautiously using, quickly moving average """
        return quickly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=None, f=fbprophet_univariate)


    def fbprophet_cautious_hypocratic(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        """ Chase residuals, somewhat cautiously using, quickly moving average """
        return quickly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=None, f=fbprophet_cautious)


    def fbprophet_exogenous_exogenous(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        return residual_chaser_factory(y=y, s=s, k=k, a=a, t=t, e=e, f1=fbprophet_exogenous, f2=fbprophet_exogenous  )


    def fbprophet_univariate_univariate(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        return residual_chaser_factory(y=y, s=s, k=k, a=a, t=t, e=e, f1=fbprophet_univariate, f2=fbprophet_univariate )


    def fbprophet_univariate_univariate_hypocratic(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        return residual_chaser_factory(y=y, s=s, k=k, a=a, t=t, e=e, f1=fbprophet_univariate, f2=fbprophet_univariate_hypocratic )


    PROPHET_SKATERS_COMPOSED = [fbprophet_exogenous_hypocratic, fbprophet_univariate_hypocratic, fbprophet_cautious_hypocratic,
                                fbprophet_exogenous_exogenous, fbprophet_univariate_univariate_hypocratic]
else:
    PROPHET_SKATERS_COMPOSED = []



if __name__ == '__main__':
    from timemachines.skatertools.data.real import hospital_with_exog
    from timemachines.skatertools.visualization.priorplot import prior_plot
    import matplotlib.pyplot as plt
    k = 1
    y, a = hospital_with_exog(k=k, n=450, offset=True)
    f = fbprophet_univariate_hypocratic
    err2 = prior_plot(f=f, k=k, y=y, n=450, n_plot=50)
    print(err2)
    plt.show()
    pass
