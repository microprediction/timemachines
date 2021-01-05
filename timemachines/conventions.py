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


def to_parameters(r:R_TYPE,bounds:BOUNDS_TYPE=None, dim:int=1):
    """ Interprets r as a set of parameter choices """
    if bounds is None:
        bounds = [ (0,1) for _ in range(dim) ]
    else:
        dim = len(bounds)

    us = ZCurveConventions().to_cube(zpercentile=r, dim=dim)      # 0 < us[i] < 1
    return [ u*(b[1]-b[0]) + b[0] for u,b in zip(us,bounds) ]


def to_parameters_exp( r:float,exponents:[float], bounds:BOUNDS_TYPE=None):
    """ Apply u->exp(e*u) to parameters """
    if bounds is not None:
        assert len(exponents)==len(bounds)
    dim = len(exponents)
    xs = to_parameters(r=r, bounds=bounds, dim=dim)
    return [ math.exp(expon*x) for expon,x in zip(exponents, xs)]

