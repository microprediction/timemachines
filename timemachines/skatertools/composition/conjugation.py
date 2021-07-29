from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, E_TYPE, T_TYPE, wrap
from typing import Any
import math


def conjugation_factory(y :Y_TYPE, s:dict, k:int, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None, r=None,
                            f=None, h=None, h_inv=None, apply_to_exog=False, h_kwargs=None, f_kwargs=None, check_inversion=False)->([float] , Any , Any):
    """

        h: R -> R
        h_inv: R -> R
    """
    y = wrap(y)
    if h_kwargs is None:
        h_kwargs = {}

    y_transformed = [yi for yi in y]
    y_transformed[0] = h(y[0], **h_kwargs)
    if check_inversion:
        y0_back = h_inv(y_transformed[0], **h_kwargs)
        assert abs(y[0]-y0_back)<0.01, 'h_inv appears not to be an inverse of h'

    if apply_to_exog:
        for k in range(1,len(y)):
            y_transformed[k] = h(y[k], **h_kwargs)

    if f_kwargs is None:
        f_kwargs = {}
    if r is not None:
        x_transformed, x_transformed_std, s = f(y=y_transformed,s=s,k=k, a=a, t=t, e=e, r=r, **f_kwargs)
    else:
        x_transformed, x_transformed_std, s = f(y=y_transformed,s=s,k=k, a=a, t=t, e=e, **f_kwargs)

    x_inv = [ h_inv(xk,**h_kwargs) for xk in x_transformed]
    shrink_factor = 0.1
    try:
        x_upper = [ h_inv(xk+shrink_factor*xk_std, **h_kwargs) for xk, xk_std in zip(x_transformed,x_transformed_std)]
        x_lower = [ h_inv(xk-shrink_factor*xk_std, **h_kwargs) for xk, xk_std in zip(x_transformed, x_transformed_std)]
        x_inv_std = [ abs(xu-xl)/(2*shrink_factor) for xu,xl in zip(x_upper, x_lower)]
    except ValueError:
        # The prediction might not remain in R+
        x_inv_std = [ 1.0 for _ in x_inv ]

    return x_inv, x_inv_std, s


def exp_conjugation_factory(y :Y_TYPE, s:dict, k:int, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None, r=None, f=None, f_kwargs=None)->([float] , Any , Any):
    """
       For when models only work on (0,inf)
       e.g. multiplicative errors etc 
    """
    assert f is not None

    if s.get('running_endog_envelope') is None:
        s['running_endog_envelope'] = 10000.

    # Maintain a rough scale that lags behind actual values and changes slowly
    y = wrap(y)
    if (abs(y[0])>2*s['running_endog_envelope']):
        s['running_endog_envelope']=1.1*s['running_endog_envelope']

    # This defines a slowly changing map into R+
    exponent = 0.1/s['running_endog_envelope']

    def h(x):
        return math.exp(exponent*x)

    def h_inv(x):
        return math.log(x)/exponent

    return conjugation_factory(y=y,s=s, k=k, a=a, t=t, e=e, r=r, f=f, h=h, h_inv=h_inv,
                               apply_to_exog=False, check_inversion=True, f_kwargs=f_kwargs)


    
    
    
    