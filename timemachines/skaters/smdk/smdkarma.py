from timemachines.skaters.smdk.smdkinclusion import using_simdkalman

if using_simdkalman:
    from timemachines.skaters.smdk.smdkarmafactory import smdk_arma_factory
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE, wrap

    def smdk_p5_d0_q3_n1000(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None):
          return smdk_arma_factory(y=y, n_agents=1000, max_p=5, max_q=3, s=s, k=k, a=a, t=t, e=e, r=r )


    def smdk_p5_d0_q3_n500(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None):
        return smdk_arma_factory(y=y, n_agents=500, max_p=5, max_q=3, s=s, k=k, a=a, t=t, e=e, r=r)


    def smdk_p5_d0_q3_n5000(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None,
                           r: R_TYPE = None):
        return smdk_arma_factory(y=y, n_agents=5000, max_p=5, max_q=3, s=s, k=k, a=a, t=t, e=e, r=r)


    SMDK_ARMA_SKATERS = [smdk_p5_d0_q3_n1000, smdk_p5_d0_q3_n500]
else:
    SMDK_ARMA_SKATERS = []



if __name__=='__main__':
    from timemachines.skatertools.data.real import hospital_with_exog
    from timemachines.skatertools.evaluation.evaluators import evaluate_mean_squared_error_with_sporadic_fit
    from timemachines.skaters.simple.thinking import thinking_fast_and_slow
    import time
    k = 1
    st = time.time()
    y, a = hospital_with_exog(k=k, n=500, offset=True)
    f = smdk_p5_d0_q3_n1000
    err = evaluate_mean_squared_error_with_sporadic_fit(f,y=y,k=k, fit_frequency=1)
    elapsed = time.time() - st
    st = time.time()
    err0 = evaluate_mean_squared_error_with_sporadic_fit(f=thinking_fast_and_slow,y=y,k=k,fit_frequency=1)
    elapsed0 = time.time()-st
    n_calcs = (len(y)-10-k)*1000
    print('Relative CPU = '+str(elapsed/elapsed0)+' with '+str(round(n_calcs/elapsed))+' agent forecasts per second for error of '+str(err/err0))
    if False:
        from timemachines.skatertools.visualization.priorplot import prior_plot
        prior_plot(f=f, k=k, y=y, n=450, n_plot=25)

