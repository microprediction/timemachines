
from timemachines.skatertools.evaluation.evaluators import evaluate_sklearn_metric_with_sporadic_fit
from momentum.functions import var_init, var_update
import random
from pprint import pprint
from timemachines.inclusion.sklearninclusion import using_sklearn
from typing import List

if using_sklearn:
    from sklearn.metrics import mean_squared_error, mean_absolute_error

# Example of comparing two or more skaters from the timemachines package


def compare(fs, ys:[float], metric=None, n_test=10, n_train=200):
    """ Keep running tally of relative errors
    :param fs: A list of skaters
    :param ys:  A very long timeseries
    :return:
    """
    if metric is None:
        metric = mean_squared_error

    n = len(ys)
    running_moments = [ var_init() for _ in fs ]

    while True:
        seq_len = n_train+n_test
        start_ndx = random.choice(list(range(n-seq_len-1)))
        y = ys[start_ndx:start_ndx + seq_len]
        es = [ evaluate_sklearn_metric_with_sporadic_fit(f=f, y=y, k=1, n_test=n_test, e_fit=60, e_nofit=-1, fit_frequency=30, metric=metric) for f in fs]
        for j,(mse,v) in enumerate(zip(es,running_moments)):
            v = var_update(v,(mse/n_test)**2)
            running_moments[j] = v

        if True:
            report = dict( sorted([ (v['mean'],f.__name__) for v,f in zip(running_moments,fs) ]) )
            print(' ')
            pprint(report)



