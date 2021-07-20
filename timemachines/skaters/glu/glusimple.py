from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE, wrap
from typing import Any
from timemachines.skatertools.components.parade import parade
from timemachines.skatertools.utilities.nonemath import nonecast
from timemachines.skatertools.ensembling.ensemblefactory import precision_weighted_ensemble_factory


def glu_simple(y :Y_TYPE, s, k:int, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None, r:R_TYPE=None):
    """ Rolling gluon
    """
    assert r is not None
    y0 = wrap(y)[0]
    if not s.get('p'):
        s = {'p':{},
             'x':y0,
             'rho':r}
        assert 0 <= s['rho'] <= 1, 'Expecting rho=r to be between 0 and 1'
    else:
        assert abs(r-s['rho'])<1e-6,'rho=r is immutable'

    if y0 is None:
        return None, s, None
    else:
        s['x'] = s['rho']*s['x'] + (1-s['rho'])*y0         # Make me better !
        x = [s['x']]*k
        _we_ignore_bias, x_std, s['p'] = parade(p=s['p'], x=x, y=y0)
        x_std_fallback = nonecast(x_std,fill_value=1.0)
        return [s['x']] * k, x_std_fallback, s
