from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE, wrap
from timemachines.skatertools.composition.residualshypocratic import quickly_moving_hypocratic_residual_factory, slowly_moving_hypocratic_residual_factory
from timemachines.skaters.simple.movingaverage import slowly_moving_average, quickly_moving_average
from timemachines.skatertools.ensembling.ensemblefactory import precision_weighted_ensemble_factory
from typing import Any

# Composition of fast and slow moving averages
# Mostly for offlinetesting, no idea if these make any sense


def thinking_slow_and_fast(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
    """
           Apply defensive quickly moving average to residuals from a slowly moving average
    """
    return quickly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, f=slowly_moving_average)


def thinking_fast_and_slow(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
    """
          Apply defensive slowly moving average to residuals from a quickly moving average
    """
    return quickly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, f=slowly_moving_average)


def thinking_slow_and_slow(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
    """
           Apply defensive slow moving average to residuals from a slowly moving average
    """
    return slowly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, f=slowly_moving_average)


def thinking_fast_and_fast(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
    """
           Apply defensive quickly moving average to residuals from a quickly moving average
    """
    return quickly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, f=quickly_moving_average)


def thinking_ensemble(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
    """
         Precision weight moving averages
    """
    fs = [ slowly_moving_average, quickly_moving_average, thinking_fast_and_fast, thinking_fast_and_slow,
           thinking_slow_and_fast, thinking_slow_and_slow ]
    return precision_weighted_ensemble_factory(fs=fs,y=y,s=s,k=k,a=a,t=t,e=e,r=0.5)


THINKING_SKATERS = [ thinking_fast_and_fast, thinking_fast_and_slow, thinking_slow_and_slow, thinking_slow_and_fast ]


if __name__=='__main__':
    k = 4
    from timemachines.skatertools.visualization.realplot import hospital_prior_plot_exogenous
    import matplotlib.pyplot as plt
    hospital_prior_plot_exogenous(f=thinking_ensemble,k=k)
    plt.show()
    pass

