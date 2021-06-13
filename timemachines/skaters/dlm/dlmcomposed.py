from timemachines.skaters.dlm.dlmunivariate import using_dlm
if using_dlm:
    from timemachines.skaters.dlm.dlmunivariate import dlm_univariate_a, dlm_univariate_b
    from timemachines.skatertools.composition.residualshypocratic import quickly_moving_hypocratic_residual_factory
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, E_TYPE, T_TYPE


    def dlm_univariate_hypocratic_a(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        return quickly_moving_hypocratic_residual_factory(f=dlm_univariate_a, y=y,s=s,k=k,a=a,t=t,e=e)


    def dlm_univariate_hypocratic_b(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        return quickly_moving_hypocratic_residual_factory(f=dlm_univariate_b, y=y,s=s,k=k,a=a,t=t,e=e)


    DLM_SKATERS_COMPOSED = [ dlm_univariate_a, dlm_univariate_b ]
else:
    DLM_SKATERS_COMPOSED = []


