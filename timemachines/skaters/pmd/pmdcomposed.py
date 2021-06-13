from timemachines.skaters.pmd.pmdinclusion import using_pmd
if using_pmd:

    from timemachines.skaters.pmd.pmdskaters import pmd_univariate
    from timemachines.skatertools.composition.residualshypocratic import quickly_moving_hypocratic_residual_factory
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, E_TYPE, T_TYPE


    def pmd_univariate_hypocratic(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        return quickly_moving_hypocratic_residual_factory(f=pmd_univariate, y=y,s=s,k=k,a=a,t=t,e=e)


    def pmd_exogenous_hypocratic(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        return quickly_moving_hypocratic_residual_factory(f=pmd_univariate, y=y,s=s,k=k,a=a,t=t,e=e)


    PMD_SKATERS_COMPOSED = [ pmd_univariate_hypocratic, pmd_exogenous_hypocratic ]
else:
    PMD_SKATERS_COMPOSED = []



