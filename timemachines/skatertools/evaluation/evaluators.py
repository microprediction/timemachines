from timemachines.skatertools.data.synthetic import brownian_with_noise, brownian_with_exogenous
from timemachines.skating import prior, residuals, prior_with_sporadic_fit
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


def evaluate_sklearn_metric(f, y, k:int, a=None, t=None, e_fit=60, e_nofit=-1, metric=None, r=None)->float:
    """ Compute prior for skater and evaluate an sklearn metric

           n_test:   Number of data points to test with

        Trains on history then computes several test samples
    """
    return evaluate_sklearn_metric_with_sporadic_fit(f=f,y=y,k=k,a=a,t=t,e_fit=e_fit, e_nofit=e_nofit, r=r, fit_frequency=1, metric=metric )


def evaluate_sklearn_metric_with_sporadic_fit(f, y, k:int, a=None, t=None, e=None, r=None, metric=None,
                                              n_test:int=10, e_fit=60,  e_nofit=-1, fit_frequency:int=100)->float:
    x, x_std = prior_with_sporadic_fit(f=f, y=y, k=k, a=a, t=t, r=r, n_test=n_test,
                                       e_fit=e_fit,  e_nofit=e_nofit, fit_frequency=fit_frequency)
    yt = targets(y)
    xk = [xt[-1] for xt in x]
    return metric(yt[-n_test:], xk[-n_test:])


def evaluate_mean_squared_error_with_sporadic_fit(f, y, k:int, a=None, t=None, r=None, n_test:int=10, e_fit=60, e_nofit=-1, fit_frequency:int=100)->float:
    return evaluate_sklearn_metric_with_sporadic_fit(f=f, y=y, k=k, a=a, t=t, r=r, metric=mean_squared_error, e_fit=e_fit, e_nofit=e_nofit, n_test=n_test, fit_frequency=fit_frequency)


def hospital_mean_square_error_with_sporadic_fit(f, k:int=1, n=120, r=None, n_test:int=10, e_fit=60, e_nofit=-1, fit_frequency:int=100)->float:
    """ Useful for a quick test of a skater, univariate and random hyper-param """
    y = hospital()[:n]
    return evaluate_mean_squared_error_with_sporadic_fit(f=f, y=y, k=k, r=r, e_fit=e_fit, n_test=n_test, e_nofit=e_nofit, fit_frequency=fit_frequency)


def hospital_mean_square_error(f, k:int=1, n=120, r=None, n_test:int=10, e_fit=60, e_nofit=-1)->float:
    """ Useful for a quick test of a skater, univariate and random hyper-param """
    return hospital_mean_square_error_with_sporadic_fit(f=f, k=k, n=n, r=r, e_fit=e_fit, n_test=n_test, e_nofit=e_nofit, fit_frequency=1)



def hospital_exog_mean_square_error_with_sporadic_fit(f, k, n=120, r=None, n_test:int=10, e_fit=60, e_nofit=-1, fit_frequency:int=100)->float:
    """ Useful for a quick test of a skater w/ exogenous inputs and known-in-advance variables """
    y, a = hospital_with_exog(n=n,k=k)
    return evaluate_mean_squared_error_with_sporadic_fit(f=f, y=y, a=a, k=k, r=r, n_test=n_test, e_fit=e_fit, e_nofit=e_nofit, fit_frequency=fit_frequency)


# Energy distance between consecutive epochs
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


EVALUATORS = [evaluate_mean_squared_error_with_sporadic_fit]