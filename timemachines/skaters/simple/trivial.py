from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE, wrap
from typing import Any

# Similar to basic skaters, but no maintenence of empirical errors is performed
# Useful as components when you just need a point estimate for some secondary quantity


def trivial_last_value(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None )->([float] , [float] , Any):
    """ Last value cache """

    if y is None:
        return None, None, s
    else:
        y0 = wrap(y)[0]       # Ignore the rest
        x = [y0]*k            # What a great prediction !
        x_std = [0.0]*k       # What a great std error estimate !
        return x, x_std, {}


def trivial_ema_r1(y :Y_TYPE, s, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None, r:R_TYPE=None):
    """ Exponential moving average
          r      weight to place on existing anchor point
    """
    assert r is not None
    y0 = wrap(y)[0]
    if not s.get('rho'):
        s = {'x':y0,
             'rho':r}
        assert 0 <= s['rho'] <= 1, 'Expecting rho=r to be between 0 and 1'
    else:
        assert abs(r-s['rho'])<1e-6,'rho=r is immutable'

    if y0 is None:
        return None, s, None
    else:
        s['x'] = s['rho']*s['x'] + (1-s['rho'])*y0         # Make me better !
        return [s['x']]*k, [1.0]*k, s


if __name__=='__main__':
    from timemachines.skatertools.data import hospital_with_exog
    from timemachines.skatertools.evaluation.evaluators import evaluate_mean_absolute_error

    k = 3
    y, a = hospital_with_exog(k=k,n=500)
    f = trivial_ema_r1
    err1 = evaluate_mean_absolute_error(f=f, k=k, y=y, a=a, r=0.9, n_burn=50)

