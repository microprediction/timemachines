from timemachines.skaters.simple.movingaverage import balanced_ema_ensemble, aggressive_ema_ensemble, \
    precision_ema_ensemble
from timemachines.skatertools.composition.residualshypocratic import quickly_moving_hypocratic_residual_factory, \
    slowly_moving_hypocratic_residual_factory
from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, E_TYPE, T_TYPE


def quick_balanced_ema_ensemble(y: Y_TYPE, s: dict, k: int = 1, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
    """
           Apply defensive quickly moving average to residuals from an EMA ensemble
    """
    return quickly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, f=balanced_ema_ensemble)


def slow_balanced_ema_ensemble(y: Y_TYPE, s: dict, k: int = 1, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
    """
           Apply defensive quickly moving average to residuals from an EMA ensemble
    """
    return slowly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, f=balanced_ema_ensemble)


def quick_aggressive_ema_ensemble(y: Y_TYPE, s: dict, k: int = 1, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
    """
           Apply defensive quickly moving average to residuals from an EMA ensemble
    """
    return quickly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, f=aggressive_ema_ensemble)


def slow_aggressive_ema_ensemble(y: Y_TYPE, s: dict, k: int = 1, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
    """
           Apply defensive slowly moving average to residuals from an EMA ensemble
    """
    return slowly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, f=aggressive_ema_ensemble)


def quick_precision_ema_ensemble(y: Y_TYPE, s: dict, k: int = 1, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
    """
           Apply defensive quickly moving average to residuals from an EMA ensemble
    """
    return quickly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, f=precision_ema_ensemble)


def slow_precision_ema_ensemble(y: Y_TYPE, s: dict, k: int = 1, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
    """
           Apply defensive slowly moving average to residuals from an EMA ensemble
    """
    return slowly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, f=precision_ema_ensemble)


HYPOCRATIC_ENSEMBLE_SKATERS = [ quick_balanced_ema_ensemble, slow_balanced_ema_ensemble,
                                quick_aggressive_ema_ensemble, slow_aggressive_ema_ensemble,
                                quick_precision_ema_ensemble, slow_precision_ema_ensemble]



if __name__=='__main__':
    k = 3
    from timemachines.skatertools.visualization.realplot import hospital_prior_plot_exogenous
    import matplotlib.pyplot as plt
    hospital_prior_plot_exogenous(f=quick_precision_ema_ensemble,k=k)
    plt.show()
    pass
