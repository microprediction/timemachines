from timemachines.skaters.sk.sfinclusion import using_statsforecast

if using_statsforecast:
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, E_TYPE, T_TYPE
    from timemachines.skatertools.composition.residualshypocratic import quickly_moving_hypocratic_residual_factory
    from timemachines.skaters.sk.sfautoarima import sf_autoarima


    def sf_autoarima_hypocratic(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        """ Chase residuals, somewhat cautiously using, quickly moving average """
        return quickly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=None, f=sf_autoarima)



SF_HYPOCRATIC_SKATERS = [ sf_autoarima_hypocratic ]
