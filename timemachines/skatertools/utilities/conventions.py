import math
from microconventions.zcurve_conventions import ZCurveConventions
from typing import List, Union, Tuple, Any

# The SKATER convention:
# ----------------------
# A time series model is a function with 7 inputs and 3 outputs
#
#        x, s, w = f( y, s, k, a, t, e, r )
#
# Some additional, optional conventions are expressed in the help functions in this file, but that
# is more for convenient.

# Inputs
Y_TYPE = Union[float, List[float]]  # Observed data, where y[0] is usually assumed to be the 'target'
S_TYPE = Any  # State previously received from callee
K_TYPE = int  # Number of steps ahead to forecast - usually integer
A_TYPE = Y_TYPE  # Known-in advance or other action-conditional variables
T_TYPE = Union[float, int]  # Epoch time of observation
E_TYPE = Union[float, int]  # Expiry in seconds (i.e. how long should callee spend computing)
E_TYPE_LIST = List[E_TYPE]
E_TYPE_OR_LIST = Union[ E_TYPE, List[E_TYPE]]
R_TYPE = float  # Hype(r) Pa(r)amete(r)s for the model


# See README.md for discussion of space-filling curves

def wrap(x):
    """ Ensure x is a list of float """
    if x is None:
        return None
    elif isinstance(x,(float,int)):
        return [float(x)]
    else:
        return list(x)


def dimension(x):
    return 0 if x is None else len(wrap(x))


def split_exogenous(y: [float])->Union[None, Tuple[float,List[float]]]:
    """ Usual convention for interpreting y
    :param y:
    :returns:  y0, exog
    """
    wy = wrap(y)
    if wy is None:
        return None
    else:
        wy = wrap(y)
        return wy[0], wy[1:] if len(wy)>=1 else None


def target(y):
    return split_exogenous(y)[0]


def targets(ys):
    return [target(y) for y in ys]


# Outputs

X_TYPE = Union[float, None]  # A point estimate, or some other anchor point deemed helpful
# S_TYPE                       Posterior state.
W_TYPE = Any  # (W)hatever else callee chooses to emit, such as a conf interval


def e_burn(n_burn=400, n=402, e_burn=-1, e_test=1000):
    """
        A vector of -1's followed by a large value
        This is typically the 'e' that will be used when sending historical data to a skater helper function

    """
    return [e_burn] * n_burn + [e_test] * (n - n_burn)




# The remainder of this module establishes space-filling curve conventions that apply to a and r

def positive_log_scale(u, low, high):
    """ Map u in (0,1) to (low,high) """
    assert 0 <= u <= 1
    assert 0 < low < high
    log_low = math.log(low)
    log_high = math.log(high)
    x = log_low + u * (log_high - log_low)
    return math.exp(x)


def to_log_space_1d(u, low, high):
    """ Approximately logarithmic map, but allows for ranges spanning zero
         returns:  float between low and high
    """
    assert 0 <= u <= 1
    assert low<high

    if 1e-8 < low < high:
        return positive_log_scale(u=u, low=low, high=high)
    elif low < -1e-8 < high < 1e-8:
        return -positive_log_scale(1 - u, low=-high, high=-low)
    elif -1e-8 < low < 1e-8 < high:
        return positive_log_scale(u=u, low=1e-8, high=high)
    elif low < -1e-8 < high < 1e-8:
        return -positive_log_scale(1 - u, low=1e-8, high=-low)
    else:
        scale = abs(high - low) / 100
        if u < 0.475:
            u1 = 1 - u / 0.475
            return -positive_log_scale(u=u1, low=scale, high=-low)
        elif 0.475 < u < 0.525:
            u2 = 20 * (u - 0.475)
            return -scale + 2 * u2 * scale
        else:
            u3 = (u - 0.525) / 0.525
            return positive_log_scale(u3, low=scale, high=high)


def implied_r(f):
    """ Dimension for search implied by a skater name """
    # Generally this isn't encouraged, as skaters might be created by functools.partial and folks
    # may forget to apply functools.upate_wrapper to preserve the __name__
    name = f if isinstance(f,str) else f.__name__
    if '_r2' in name:
        return 2
    elif '_r3' in name:
        return 3
    elif '_r1' in name:
        return 1
    else:
        return 0


BOUNDS_TYPE = List[Union[Tuple, List]]  # scipy.optimize style bounds [ (low,high), (low, high),... ]


def to_space(p: float, bounds: BOUNDS_TYPE = None, dim: int = 1):
    """ Interprets p as a point in a rectangle in R^2 or R^3

         :param bounds  [ (low,high), (low,high), (low,high) ] defaults to unit cube
         :param dim     Dimension. Only used if bounds are not supplied.

    """
    if bounds is None:
        bounds = [(0, 1) for _ in range(dim)]
    else:
        dim = len(bounds)

    if dim>1:
        us = reversed(ZCurveConventions().to_cube(zpercentile=p, dim=dim))  # 0 < us[i] < 1
    else:
        us = [p]
    return [u * (b[1] - b[0]) + b[0] for u, b in zip(us, bounds)]


def from_space(ps: [float], bounds: BOUNDS_TYPE=None)->float:
    """ [ , ]^n -> [0,1] """
    if bounds is None:
        bounds = [(0, 1) for _ in range(len(ps))]
    us = [(pi - b[0]) / (b[1] - b[0]) for pi, b in zip(ps, bounds)]
    for u in us:
        assert 0 <= u <= 1, "bounds are inconsistent with p=" + str(ps)
    if len(us)>1:
        return ZCurveConventions().from_cube(list(reversed(us)))
    else:
        return us[0]


def to_log_space(p:float, bounds: BOUNDS_TYPE):
    """ Interprets p as a point in a rectangle in R^2 or R^3 using Morton space-filling curve

            :param bounds  [ (low,high), (low,high), (low,high) ] defaults to unit cube
            :param dim     Dimension. Only used if bounds are not supplied.

       Very similar to "to_space" but assumes speed varies with logarithm
       """
    assert 0 <= p <= 1
    dim = len(bounds)
    us = list(reversed(ZCurveConventions().to_cube(zpercentile=p, dim=dim)))  # 0 < us[i] < 1
    return [to_log_space_1d(u, low=b[0], high=b[1]) for u, b in zip(us, bounds)]


def to_int_log_space(p: float, bounds: BOUNDS_TYPE):
    """ Interprets p as a point in an integer lattice in R^2 or R^3 using Morton space-filling curve, integers only

            :param bounds  [ (low,high), (low,high), (low,high) ] defaults to unit cube

       Very similar to "to_space" but assumes speed varies with logarithm
       """
    assert 0 <= p <= 1
    prms = to_log_space(p=p, bounds=bounds)
    return [int(prm) for prm in prms]
