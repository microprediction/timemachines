from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, E_TYPE, T_TYPE
from typing import Any
from timemachines.skatertools.composition.residualcomposition import residual_chaser_factory
from timemachines.skaters.simple.hypocratic import quickly_hypocratic, slowly_hypocratic


def quickly_moving_hypocratic_residual_factory(y :Y_TYPE, s:dict, k:int, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None,
                                               f=None, r=None)->([float] , Any , Any):
    """ A simple way to convert a skater into one whose residuals are cautiously chased with a quickly moving average

           f : skater
           r : skater's hyper-params, if any

    """
    return residual_chaser_factory(y=y,s=s,k=k,a=a,t=t,e=e,f1=f,r1=r,f2=quickly_hypocratic,r2=r)


def slowly_moving_hypocratic_residual_factory(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                              e: E_TYPE = None,
                                              f=None, r=None) -> ([float], Any, Any):
    """ A simple way to convert a skater into one whose residuals are cautiously chased with a slowly moving average

           f : skater
           r : skater's hyper-params, if any

    """
    return residual_chaser_factory(y=y, s=s, k=k, a=a, t=t, e=e, f1=f, r1=r, f2=slowly_hypocratic, r2=r)