# Hyper-parameter conventions
import math
import numpy as np
from microconventions.zcurve_conventions import ZCurveConventions
from typing import List, Union, Callable, Tuple, Any
from sklearn.metrics import mean_squared_error

# The SKATER conventions
Y_TYPE = Union[float,List[float]]
S_TYPE = Any
K_TYPE = Union[float,int]
A_TYPE = Y_TYPE
T_TYPE = Union[float,int]
E_TYPE = Union[float,int]
R_TYPE = float

BOUNDS_TYPE = List[Union[Tuple,List]]   # scipy.optimize style bounds [ (low,high), (low, high),... ]

# Intended usage:


def posterior(f:Callable,         # "Model" to run forward through observations
             ys:[Y_TYPE],         # Observations
             k:K_TYPE,            # Steps ahead to predict
             ats:[A_TYPE]=None,   # Data known in advance
             ts:[T_TYPE]=None,    # Times of observations (epoch seconds)
             e:E_TYPE=None,       # Computation time limit per observation
             r:R_TYPE=0.5):       # Hyper-parameters
    """ Compute k-step ahead estimates """
    s = None
    ats = [None for _ in ys] if ats is None else ats
    ts = [None for _ in ys] if ts is None else ts
    xs = list()
    for y, t, a in zip(ys, ats, ts):
        x, s = f(y=y, s=s, k=k, a=a, t=t, e=e, r=r)
        xs.append(x)
    return xs


def prior(f, ys, k:int, ats=None, ts=None, e=None, r=0.5, x0=np.nan):
    """ Compute k-step ahead estimates aligned for easy comparison against actual """
    # Note k must be integer
    xs = posterior(f=f, ys=ys[:-k], k=k, ats=ats, ts=ts, e=e, r=r)
    return [x0]*k + xs


def brownian(n):
    ys_ = np.cumsum(np.random.randn(n))
    ys = [y + np.random.randn() for y in ys_]
    return list(ys)


def prior_plot(f, ys=None, k=None, ats=None, ts=None, e=None, r=0.5, x0=np.nan, n=150, n_plot=25):
    if ys is None:
       ys = brownian(n=n)
    xs = prior(f=f,ys=ys,k=k,ats=ats,ts=ts,e=e,r=r,x0=x0)
    import matplotlib.pyplot as plt
    if ts is None:
        ts = range(len(ys))

    plt.plot(ts[-n_plot:],ys[-n_plot:],'b*')
    plt.plot(ts[-n_plot:],xs[-n_plot:],'g-')
    plt.legend(['Data','Prediction'])
    plt.show()


def rmse1(f,ys=None,k=1,n=200):
    """ Useful for a quick test """
    if ys is None:
       ys = brownian(n=n)
    xs = prior(f=f,ys=ys,k=1,ats=None, ts=None)
    rmse = mean_squared_error(ys[k:], xs[k:], squared=False)
    return rmse


def positive_log_scale(u,low,high):
    assert 0 < low < high
    log_low = math.log(low)
    log_high = math.log(high)
    x = log_low + u * (log_high - log_low)
    return math.exp(x)


def balanced_logscale(u, low, high):
    """ Approximately logarithmic map, but allows for ranges spanning zero """
    # Median at zero

    if 1e-8 < low < high:
        return positive_log_scale(u=u,low=low,high=high)
    elif low < -1e-8 < 1e8 < high:
        return -positive_log_scale(1-u,low=-high,high=-low)
    else:
        scale = abs(high - low) / 100
        if u<0.475:
            u1 = 1-u/0.475
            return -positive_log_scale(u=u1,low=scale,high=-low)
        elif 0.475<u<0.525:
            u2 = 20*(u-0.475)
            return -scale+2*u2*scale
        else:
            u3 = (u-0.525)/0.525
            return positive_log_scale(u3,low=scale,high=high)


def to_parameters(r:R_TYPE,bounds:BOUNDS_TYPE=None, dim:int=1):
    """ Interprets r as a set of parameter choices
        We reverse the pymorton convention so that the first listed
        parameter is the most important one (it varies more smoothly as r varies)
    """
    if bounds is None:
        bounds = [ (0,1) for _ in range(dim) ]
    else:
        dim = len(bounds)

    us = reversed( ZCurveConventions().to_cube(zpercentile=r, dim=dim) )     # 0 < us[i] < 1
    return [ u*(b[1]-b[0])+b[0] for u, b in zip(us, bounds)]


def to_parameters_logscale(r:float, bounds:BOUNDS_TYPE):
    """ Apply u->exp(e*u) to parameters """
    assert 0<=r<=1
    dim = len(bounds)
    us = reversed( ZCurveConventions().to_cube(zpercentile=r, dim=dim) )      # 0 < us[i] < 1
    return [ balanced_logscale(u,low=b[0],high=b[1]) for u, b in zip(us, bounds) ]

