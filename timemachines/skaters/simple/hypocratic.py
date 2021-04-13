from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE
from timemachines.skaters.simple.movingaverage import empirical_ema_r1
from timemachines.skatertools.utilities.conventions import wrap

# Skaters that shrink towards zero


def hypocratic_ema_r1(y :Y_TYPE, s, k:int, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None, r:R_TYPE=None):
    """
         r :  moving average parameter  (e.g. 0.75 is fast, 0.95 is slow)
    """
    y0 = wrap(y)[0]
    assert r is not None
    x, x_std, s = empirical_ema_r1(y=y0,s=s,k=k,a=a,t=t,e=e,r=r)

    def hypocratic(x:float, x_std:float, confidence=0.5):
        """ Shrink residual prediction towards zero """
        import math
        if abs(x_std)<1e-6 or abs(x)<1e-3*x_std:
            return 0.0
        else:
            return x*math.tanh(confidence*abs(x)/(3*x_std))

    x_resid = [ hypocratic(xi,x_std) for xi,x_std in zip(x,x_std) ]
    return x_resid, x_std, s


def slowly_hypocratic(y :Y_TYPE, s, k:int, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None):
    return hypocratic_ema_r1(y=y,s=s,k=k,a=a,t=t,e=e,r=0.95)


def quickly_hypocratic(y :Y_TYPE, s, k:int, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None):
    return hypocratic_ema_r1(y=y,s=s,k=k,a=a,t=t,e=e,r=0.75)


if __name__=='__main__':
    from timemachines.skatertools.data.real import hospital_with_exog
    from timemachines.skatertools.evaluation.evaluators import evaluate_mean_absolute_error

    k = 3
    y, a = hospital_with_exog(k=k,n=500)
    f = slowly_hypocratic
    err1 = evaluate_mean_absolute_error(f=f, k=k, y=y, a=a, n_burn=50)
