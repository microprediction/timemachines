from timemachines.skatertools.data.synthetic import brownian_with_noise, brownian_with_exogenous
from timemachines.skating import prior, residuals
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import energy_distance
import numpy as np
from timemachines.skatertools.utilities.conventions import targets
from typing import List
from timemachines.skatertools.data.real import hospital, hospital_with_exog

# Evaluation of skaters


def evaluator_from_name(name):
    valid = [f for f in EVALUATORS if f.__name__==name ]
    return valid[0] if len(valid)==1 else None


def evaluate_sklearn_metric(f, y, k:int, a=None, t=None, e=None, r=None, metric=None, n_burn=None)->float:
    """ Compute prior for skater and evaluate an sklearn metric """
    assert metric is not None
    assert n_burn is not None
    x, x_std = prior(f=f, y=y, k=k, a=a, t=t, r=r, e=e )
    yt = targets(y)
    xk = [ xt[-1] for xt in x ]  # k-steps ahead
    return metric(yt[n_burn:], xk[n_burn:] )


def evaluate_mean_squared_error(f, y, k:int, a=None, t=None, e=None, r=None, n_burn=None)->float:
    assert n_burn is not None
    return evaluate_sklearn_metric(f=f, y=y, k=k, a=a, t=t, e=e, r=r, metric=mean_squared_error, n_burn=n_burn)


def evaluate_mean_absolute_error(f, y, k:int, a=None, t=None, e=None, r=None, n_burn=None )->float:
    assert n_burn is not None
    return evaluate_sklearn_metric(f=f, y=y, k=k, a=a, t=t, e=e, r=r, metric=mean_absolute_error, n_burn=n_burn )


def hospital_mean_square_error(f, k:int=1, n=120, n_burn=30, r=None)->float:
    """ Useful for a quick test of a skater, univariate and random hyper-param """
    y = hospital()[:n]
    return evaluate_mean_squared_error(f=f, y=y, k=k, r=r, n_burn=n_burn)


def hospital_exog_mean_square_error(f, k, n=120, n_burn=30, r=None)->float:
    """ Useful for a quick test of a skater w/ exogenous inputs and known-in-advance variables """
    y, a = hospital_with_exog(n=n,k=k)
    return evaluate_mean_squared_error(f=f, y=y, a=a, k=k, r=r, n_burn=n_burn)


# Energy distance between simple of consecutive epochs
# (a more speculative way to evaluate point estimates)


def chunk_to_end(l:List, n:int)-> List[List]:
    """ Break list in to evenly sized chunks
        :param n: Size of batches
    """
    rl = list(reversed(l))
    chunks = [ list(reversed(rl[x:x + n])) for x in range(0, len(rl), n) ]
    return list(reversed(chunks[:-1]))


def evaluate_energy(f, y=None, k=1, a=None, t=None, e=None, r=None, n_burn=30, n_epoch=30):
    r = residuals(f=f, y=y, k=k, a=a, t=t, e=e, r=r, n_burn=n_burn)
    r_chunks = chunk_to_end(r,n_epoch)
    assert len(r_chunks)>=2,'Cannot evaluate ... try a shorter n_epoch '
    uv = [ (u,v) for u,v in zip( r_chunks[1:],r_chunks[:-1] )]   # residuals in one epoch versus the next
    return np.mean([ energy_distance(u_values=u_values,v_values=v_values) for u_values, v_values in uv ])


def hospital_energy(f, k=3, n=100, n_epoch=20, n_burn=18):
    y, a = hospital_with_exog(n=n+5)
    return evaluate_energy(f=f, y=y, k=k, a=a, n_burn=n_burn, n_epoch=n_epoch)


def brownian_energy(f, n=500, **kwargs):
    ys = brownian_with_noise(n)
    return evaluate_energy(f=f, y=ys, **kwargs)


def exogenous_energy(f, n=500, **kwargs):
    ys = brownian_with_exogenous(n)
    return evaluate_energy(f=f, y=ys, **kwargs)


EVALUATORS = [ evaluate_mean_squared_error, evaluate_mean_absolute_error ]