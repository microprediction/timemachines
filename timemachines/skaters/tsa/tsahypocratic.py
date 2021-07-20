from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE
from timemachines.skaters.tsa.tsaconstant import TSA_D0_SKATERS
from timemachines.skatertools.ensembling.ensemblefactory import precision_weighted_ensemble_factory
from timemachines.skatertools.utilities.conventions import wrap


def hypocratic_tsa_d0_factory(y :Y_TYPE, s, k:int, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None, r:R_TYPE=None):
    """
         r :  moving average parameter  (e.g. 0.75 is fast, 0.95 is slow)
    """
    y0 = wrap(y)[0]
    assert r is not None
    x, x_std, s = precision_weighted_ensemble_factory(fs=TSA_D0_SKATERS, y=y0, s=s, k=k, a=a, t=t, e=e, r=r)

    def hypocratic(x:float, x_std:float, confidence=0.5):
        """ Shrink residual prediction towards zero """
        # Remains to be seen if this is sensible for TSA residuals
        import math
        if abs(x_std)<1e-6 or abs(x)<1e-3*x_std:
            return 0.0
        else:
            return x*math.tanh(confidence*abs(x)/(3*x_std))

    x_resid = [ hypocratic(xi,x_std) for xi,x_std in zip(x,x_std) ]
    return x_resid, x_std, s


def tsa_slowly_hypocratic_d0_ensemble(y :Y_TYPE, s, k:int, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None):
    return hypocratic_tsa_d0_factory(y=y,s=s,k=k,a=a,t=t,e=e,r=0.95)


def tsa_quickly_hypocratic_d0_ensemble(y :Y_TYPE, s, k:int, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None):
    return hypocratic_tsa_d0_factory(y=y,s=s,k=k,a=a,t=t,e=e,r=0.75)


TSA_DO_HYPOCRATIC_SKATERS = [tsa_slowly_hypocratic_d0_ensemble, tsa_quickly_hypocratic_d0_ensemble]