from momentum import rvar  # Could easily add skew, kurtosis
from typing import Union, SupportsFloat, List


# A "parade" is a procession of l-step ahead predictions that are waiting to be judged when data arrives.
# The predictions are stored in an array where the n'th entry will be judged after n data points arrive.
# There are multiple entries in each row of the parade. For example if data is sampled on the hour,
# then a 3-hour ahead forecast made at 11:01 am will march alongside a 5-hour ahead prediction made two hours
# earlier at 9:00am. Both will be judged at 2pm. However they are tagged with the prediction horizon, allowing
# for rolling statistics to be tracked separately for 1-hr, 2-hr,...10-hr ahead predictions, say.
#
# The usage is pretty simple. Send target y and posterior predictions (i.e. those using information y) to
# parade_update, then  use parade_mean, parade_std. See tsaconstant or simple.movingaverage for examples.


def parade(p: dict, x: Union[List[SupportsFloat], None], y: Union[SupportsFloat, None], rho=0.01):
    """
          A 'parade' holds previous predictions and truths, and can be used to determine a running
          estimate of the empirical errors of the predictions. The usage is

          p=parade({})
          ... do something to create predictions x k-steps ahead when you receive y ...
          p = parade(p=p,x=x,y=y)     # submit the prediction vector and the observation y

    :param p:   state - supply empty dict on first call
    :param y:   incoming observation
    :param x:   term structure of predictions out k steps ahead, made after y received
    :param rho: Recency weighting for the empirical errors
    returns:  mean, std, of model residuals and the posterior state s'

    A special convention allows the caller to reset the empirical moments. Pass x=None and y=None

    """
    # Initialize
    if not p:
        k = len(x)
        p = {'predictions': [[] for _ in range(k)],  # Holds the cavalcade
             'moments': [rvar({}, rho=rho, n=5) for _ in range(k)]}  # Could use kurtosis_init here for more moment
    else:
        assert len(x) == len(p['predictions'])  # 'k' is immutable

    if x is None and y is None:
        # This will "reset" the running moments, but keep the existing store of predictions and observations
        p_mean, p_std = parade_mean(p), parade_std(p)
        p['moments'] = [rvar(rho=rho, n=5) for _ in range(k)]
        return p_mean, p_std, p
    else:
        assessable = p['predictions'].pop(0)
        if assessable:
            for j, xi in assessable:
                p['moments'][j] = rvar(p['moments'][j], y - xi)

        p['predictions'].append(list())
        for j, xj in enumerate(x):
            p['predictions'][j].append((j, xj))

        return parade_mean(p), parade_std(p), p


# Helpers for extracting skater-ready predictions

def parade_bias(p):
    """ Model bias is negative of mean residual """
    return [mj.get('mean') for mj in p['moments']]


def parade_mean(p):
    """ Note the sign, E[y-f()] """
    return [noneneg(mj.get('mean')) for mj in p['moments']]


def parade_std(p):
    return [mj.get('std') for mj in p['moments']]


def noneneg(x):
    return -x if x is not None else None


if __name__ == '__main__':
    from pprint import pprint
    import numpy as np

    y = list(range(100))
    x = [yi - 0.5 + np.random.randn() for yi in y][1:]
    p = {}
    for xi, yi in zip(x, y):
        _, _, p = parade(p=p, x=[xi] * 3, y=yi)
    pprint(p)

    print(parade_mean(p=p))
    print(parade_std(p=p))
