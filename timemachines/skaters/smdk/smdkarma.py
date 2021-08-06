from timemachines.skaters.smdk.smdkarmafactory import smdk_arma_factory
from timemachines.skaters.smdk.smdkinclusion import using_simdkalman

if using_simdkalman:

    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE, wrap

    def smdk_p5_d0_q3_n50(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None):
          return smdk_arma_factory(y=y, n_agents=50, max_p=5, max_q=3, s=s, k=k, a=a, t=t, e=e, r=r )


    def smdk_p5_d0_q3_n500(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None):
        return smdk_arma_factory(y=y, n_agents=500, max_p=5, max_q=3, s=s, k=k, a=a, t=t, e=e, r=r)


    SMDK_ARMA_SKATERS = [ smdk_p5_d0_q3_n50, smdk_p5_d0_q3_n500 ]
else:
    SMDK_ARMA_SKATERS = []



if __name__=='__main__':
    from timemachines.skatertools.data.real import hospital_with_exog
    from timemachines.skatertools.visualization.priorplot import prior_plot
    import time
    k = 1
    y, a = hospital_with_exog(k=k, n=450, offset=True)
    f = smdk_p5_d0_q3_n500
    err2 = prior_plot(f=f, k=k, y=y, n=450, n_plot=25)
    pass

