from typing import Callable
from timemachines.conventions import Y_TYPE, K_TYPE, A_TYPE, T_TYPE, E_TYPE, R_TYPE
import numpy as np


def posterior(f:Callable,         # "Model" to run forward through observations
             ys:[Y_TYPE],         # Observations
             k:K_TYPE,            # Steps ahead to predict, typically integer
             ats:[A_TYPE]=None,   # Data known in advance, or maybe action-conditional
             ts:[T_TYPE]=None,    # Times of observations (epoch seconds)
             e:E_TYPE=None,       # Computation time limit per observation
             r:R_TYPE=0.5):       # Hype(r)-pa(r)amete(r)s
    """ Compute k-step ahead estimates """
    s = None
    ats = [None for _ in ys] if ats is None else ats
    ts = [None for _ in ys] if ts is None else ts
    xs = list()
    for y, t, a in zip(ys, ats, ts):
        x, s, _ = f(y=y, s=s, k=k, a=a, t=t, e=e, r=r)
        xs.append(x)
    return xs


def prior(f, ys, k:int, ats=None, ts=None, e=None, r=0.5, x0=np.nan):
    """ Compute k-step ahead estimates aligned for easy comparison against actual """
    # Note k must be integer
    xs = posterior(f=f, ys=ys[:-k], k=k, ats=ats, ts=ts, e=e, r=r)
    return [x0]*k + xs
