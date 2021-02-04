from typing import Callable
from timemachines.skaters.conventions import Y_TYPE, K_TYPE, A_TYPE, T_TYPE, E_TYPE, R_TYPE, targets
import numpy as np


# Basic iteration of state machines implied by skater functions

def posterior(f:Callable,       # "Model" to run forward through observations
              y:[Y_TYPE],       # Observations
              k:K_TYPE=1,       # Steps ahead to predict, typically integer
              a:[A_TYPE]=None,  # Data known in advance, or maybe action-conditional
              t:[T_TYPE]=None,  # Times of observations (epoch seconds)
              e:E_TYPE=None,    # Computation time limit per observation, and fit hint
              r:R_TYPE=None):   # Hype(r)-pa(r)amete(r)s

    """ Compute all k-step ahead estimates given history
         returns:  x:     [ [float] ]
                   x_std: [ [float] ]
    """
    s = dict()
    a = [None for _ in y] if a is None else a
    t = [None for _ in y] if t is None else t
    if r is not None:
        assert 0 <= r <= 1
    x = list()
    x_std = list()
    for yi, ai, ti in list(zip(y, a, t)):
        if r is None:
            xi, x_stdi, s = f(y=yi, s=s, k=k, a=ai, t=ti, e=e)
        else:
            xi, x_stdi, s = f(y=yi, s=s, k=k, a=ai, t=ti, e=e, r=r)
        x.append(xi)
        x_std.append(x_stdi)
    return x, x_std


def prior(f, y:[Y_TYPE], k:int, a:[A_TYPE]=None, t:[T_TYPE]=None, e:E_TYPE=None, r:float=None,
          x0:float=np.nan, std0:float=np.nan):
    """ Compute k-step ahead estimates
        Align with y for easy comparison against actual
    """
    assert k is not None
    x_post, x_std = posterior(f=f, y=y, k=k, a=a, t=t, e=e, r=r)
    x_prior = [[x0]*k]*k + list(x_post)[:-k]
    x_prior_std = [[std0]*k]*k + list(x_post)[:-k]
    return x_prior, x_prior_std


def residuals(f, y:[Y_TYPE], k:int, a=None, t=None, e=None, r=0.5, x0=np.nan, n_burn=50):
    assert n_burn>k
    x, _ = prior(f=f, y=y, k=k, a=a, t=t, e=e, r=r, x0=x0)
    yt = targets(y)
    xk = [ xt[-1] for xt in x]
    return np.array(xk[-n_burn:])-np.array(yt[-n_burn:])
