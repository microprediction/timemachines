from timemachines.synthetic import brownian_with_noise, brownian_with_exogenous
from timemachines.skating import prior, residuals
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import energy_distance
import numpy as np
from timemachines.conventions import targets
from typing import List




# Evaluation of skaters
#   (pretty standard scoring rules)

def evaluate_sklearn_metric(f, ys, metric, n_burn, **kwargs):
    """ Useful for a quick test """
    xs = prior(f=f,ys=ys,**kwargs)
    yt = targets(ys)
    return metric(yt[n_burn:], xs[n_burn:])


def evaluate_mean_squared_error(f, ys, n_burn, **kwargs):
    return evaluate_sklearn_metric(f=f, ys=ys, metric=mean_squared_error, n_burn=n_burn, **kwargs)


def evaluate_mean_absolute_error(f, ys, n_burn, **kwargs):
    return evaluate_sklearn_metric(f=f, ys=ys, metric=mean_absolute_error, n_burn=n_burn, **kwargs)


def evaluate_mean_absolute_percentage_error(f, ys, n_burn, **kwargs):
    return evaluate_sklearn_metric(f=f, ys=ys, metric=mean_absolute_percentage_error(), n_burn=n_burn, **kwargs)


def quick_brown_fox(f, n=120, n_burn=30,  **kwargs):
    """ Useful for a quick test of a skater, w/o exogenous inputs """
    ys = brownian_with_noise(n=n)
    return evaluate_mean_squared_error(f=f,ys=ys, n_burn=n_burn, **kwargs)


def lazy_dog(f, n=120, n_burn=30, **kwargs):
    """ Useful for a quick test of a skater, w/ exogenous inputs """
    ys = brownian_with_exogenous(n=n)
    return evaluate_mean_squared_error(f=f,ys=ys, n_burn=n_burn, **kwargs)


def quick_brown_fox_randomized(f, n=200, n_burn=30,  **kwargs):
    """ Useful for a quick test of a skater, w/o exogenous inputs """
    r = np.random.rand(1)
    ys = brownian_with_noise(n=n)
    try:
        rmse = evaluate_mean_squared_error(f=f,ys=ys, n_burn=n_burn, r=r, **kwargs)
    except:
        raise('Error running '+f.__name__ + ' with r='+str(r))


# Energy distance between residuals of consecutive epochs
# (a more speculative way to evaluate point estimates)

def chunk_to_end(l:List, n:int)-> List[List]:
    """
        :param n: Size of batches
    """
    rl = list(reversed(l))
    chunks = [ list(reversed(rl[x:x + n])) for x in range(0, len(rl), n) ]
    return list(reversed(chunks[:-1]))


def evaluate_energy(f, ys=None, k=1, ats=None, ts=None, e=None, r=0.5, n_burn=50, n_epoch=100):
    r = residuals(f=f, ys=ys, k=k, ats=ats, ts=ts, e=e, r=r, n_burn=n_burn)
    r_chunks = chunk_to_end(r,n_epoch)
    return np.mean([ energy_distance(u_values=u_values,v_values=v_values) for u_values, v_values in zip( r_chunks[1:],r_chunks[:-1] )])


def brownian_energy(f, n=500, **kwargs):
    ys = brownian_with_noise(n)
    return evaluate_energy(f=f,ys=ys, **kwargs)


def exogenous_energy(f, n=500, **kwargs):
    ys = brownian_with_exogenous(n)
    return evaluate_energy(f=f,ys=ys, **kwargs)


