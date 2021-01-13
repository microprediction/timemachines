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
A_TYPE = float  # Known-in advance or other action-conditional variables
# See README.md for discussion of space-filling curves
T_TYPE = Union[float, int]  # Epoch time of observation
E_TYPE = Union[float, int]  # Expiry in seconds (i.e. how long should callee spend computing)
R_TYPE = float  # Hype(r) Pa(r)amete(r)s for the model


# Again, see README.md for discussion of space-filling curves


# Expresses the y convention
def target(y):
    try:
        return y[0]
    except:
        return y


def targets(ys):
    try:
        return [y[0] for y in ys]
    except:
        return [y for y in ys]


# Outputs

X_TYPE = Union[float, None]  # A point estimate, or some other anchor point deemed helpful
# S_TYPE                       Posterior state.
W_TYPE = Any  # (W)hatever else callee chooses to emit, such as a conf interval


def separate_observations(y: Y_TYPE, dim: int):
    """ Usual convention for interpreting y, and checking dimension
    :param s:
    :param y:
    :returns:  y0, exog
    """
    if dim == 1:
        y0 = y
        exog = None
    else:
        y0 = y[0]
        exog = [y[1:]]
    return y0, exog


def dimension(y):
    try:
        return len(y)
    except:
        return 1


def initialize_buffers(s, y: Y_TYPE):
    s['dim'] = dimension(y)
    s['buffer'] = list()  # Target
    if s['dim'] > 1:
        s['exogenous'] = list()  # Exogenous
    s['model'] = None
    s['advance'] = list()  # Variables known in advance
    s['staleness'] = 0
    return s


def update_buffers(s, a: A_TYPE, exog: [float], y0: float):
    # Store "target" and other observations or vars known in advance
    s['buffer'].append(y0)
    if exog is not None:
        s['exogenous'].append(exog[0])
    if a is not None:
        s['advance'].append(a)
    return s


# The remainder of this module establishes space-filling curve conventions that apply to a and r

def positive_log_scale(u, low, high):
    """ Map u in (0,1) to (low,high) """
    assert 0 < low < high
    log_low = math.log(low)
    log_high = math.log(high)
    x = log_low + u * (log_high - log_low)
    return math.exp(x)


def to_log_space_1d(u, low, high):
    """ Approximately logarithmic map, but allows for ranges spanning zero """
    # Avoid singularity at zero

    if 1e-8 < low < high:
        return positive_log_scale(u=u, low=low, high=high)
    elif low < -1e-8 < 1e-8 < high:
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

    us = reversed(ZCurveConventions().to_cube(zpercentile=p, dim=dim))  # 0 < us[i] < 1
    return [u * (b[1] - b[0]) + b[0] for u, b in zip(us, bounds)]


def from_space(ps: [float], bounds: BOUNDS_TYPE=None):
    """ [ , ]^n -> [0,1] """
    if bounds is None:
        bounds = [(0, 1) for _ in range(len(ps))]
    us = [(pi - b[0]) / (b[1] - b[0]) for pi, b in zip(ps, bounds)]
    for u in us:
        assert 0 <= u <= 1, "bounds are inconsistent with p=" + str(ps)
    return ZCurveConventions().from_cube(list(reversed(us)))


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
