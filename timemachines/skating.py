from typing import Callable
from timemachines.skatertools.utilities.conventions import Y_TYPE, K_TYPE, A_TYPE, T_TYPE, E_TYPE,\
    E_TYPE_OR_LIST, R_TYPE, targets
import numpy as np
import math


###########################################################################
#                                                                         #
#  Sequential assimilation of observations using a skater, strongly       #
#  suggesting what a skater function is supposed to accomplish, and how   #
#  it should behave                                                       #
#                                                                         #
#  See also skatertools.evaluation.evaluators                             #
#           skatertools.visualization.priorplot.prior_plot                #
#                                                                         #
###########################################################################
from typing import Union


def posterior(f:Callable,       # "Model" to run forward through observations
              y:[Y_TYPE],       # Observations
              k:K_TYPE=1,       # Steps ahead to predict, typically integer
              a:[A_TYPE]=None,  # Data known in advance, or maybe action-conditional variables
              t:[T_TYPE]=None,  # Times of observations (epoch seconds)
              e:E_TYPE_OR_LIST=None,  # Computation time limit per observation during 'fast' stage
              r:R_TYPE=None):   # Optional, will alter broadcast of e, if e is scalar or None

    """ Sequentially feed all data to skater and accumulate predictions

         returns:  x:     [ [float] ]    list of all prediction vectors
                   x_std: [ [float] ]    list of all std vectors
    """
    assert k<len(y)

    # Broadcasting
    a = [None for _ in y] if a is None else a
    t = [None for _ in y] if t is None else t
    e = [None for _ in y] if e is None else e


    # Hyper-param must be in [0,1]
    if r is not None:
        assert 0 <= r <= 1

    # Feed one data point at a time to the skater ...
    s = dict()  # <- Skater will receive empty dictionary on the first invocation
    x = list()
    x_std = list()
    for yi, ai, ti, ei in list(zip(y, a, t, e)):
        if r is None:
            xi, x_stdi, s = f(y=yi, s=s, k=k, a=ai, t=ti, e=ei)
        else:
            xi, x_stdi, s = f(y=yi, s=s, k=k, a=ai, t=ti, e=ei, r=r)
        x.append(xi)
        x_std.append(x_stdi)
    return x, x_std


def prior(f, y:[Y_TYPE], k:int, a:[A_TYPE]=None, t:[T_TYPE]=None, e:E_TYPE_OR_LIST=None, r:float=None,
          x0:float=np.nan, std0:float=np.nan):
    """ Compute k-step ahead estimates then align predictions with the observations y, for
        easy error computation. Pre-pad with naive forecast. Ignore last k data points.
    """
    x_post, x_std = posterior(f=f, y=y[:-k], k=k, a=a, t=t, e=e, r=r)
    x_prior = [[x0]*k]*k + list(x_post)
    x_prior_std = [[std0]*k]*k + list(x_post)
    return x_prior, x_prior_std


def prior_with_sporadic_fit(f, y:[Y_TYPE], k:int, a=None, t=None, e_fit=60,  e_nofit=-1, fit_frequency:int=100, r=0.5, x0=np.nan, n_test:int=10):
    """ Similar to the above, but we send a pattern like
         e = [ -1 .. -1 60 -1 .. -1 60 -1 ... -1, 60 60 60 60 60 60 60 ] to reduce computation time during a burn-in period

           e_fit - Value of e to provide when fit is desired (written as E above)
           train_frequency - How often to suggest fit during burn-in period
           n_test - The number of data points for which the skater is instructed to "do something that might take a while" (i.e. e>0 )

    """
    n_reps = math.ceil(len(y) / fit_frequency)
    e = (([e_nofit] * (fit_frequency - 1) + [e_fit]) * n_reps)[:len(y) - k - n_test] + [e_fit] * n_test
    return prior(f=f, y=y, k=k, a=a, t=t, e=e, r=r, x0=x0)


def residuals(f, y:[Y_TYPE], k:int, a=None, t=None, e=None, r=0.5, x0=np.nan, n_burn=50):
    """ Feed fast skater all data points, then report residuals """
    assert n_burn>k
    x, _ = prior(f=f, y=y, k=k, a=a, t=t, e=e, r=r, x0=x0)
    yt = targets(y)
    xk = [ xt[-1] for xt in x]
    return np.array(xk[-n_burn:])-np.array(yt[-n_burn:])

