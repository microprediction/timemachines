import numpy as np
from timemachines.skatertools.combining.hubermean import huber_mean

# Skaters output [ xs ], [ x_stds ]
# The utils here combining predictions
# See also the pattern in ensembling/precisionweightedskater where
# the convention is to pass x,x_std in interleaved fashion instead.


def combine_using_mean(xs,x_stds):
    """ Combine forecasts
    :param xs:      List of forecast vectors, one for each model
    :param x_stds:  List of forecast std vectors, one for each model
    :return:
    """
    n_models = len(xs) if isinstance(xs,list) else np.shape(xs)[0]
    if n_models==1:
        return xs[0], x_stds[0]
    else:
        avg_x = list(np.average(np.array(xs), axis=0))
        avg_xstd = list(np.average(np.array(x_stds), axis=0)) # lazy and not particularly strongy motivated :)
        return avg_x, avg_xstd


def combine_using_median(xs:[[float]],x_stds:[[float]]):
    n_models = len(xs) if isinstance(xs, list) else np.shape(xs)[0]
    if n_models == 1:
        return xs[0], x_stds[0]
    else:
        avg_x = list(np.nanmedian(np.array(xs), axis=0))
        avg_xstd = list(np.nanmedian(np.array(x_stds), axis=0))
        return avg_x, avg_xstd


def combine_using_huber(xs, x_stds, **huber_kwargs):
    n_models = len(xs) if isinstance(xs, list) else np.shape(xs)[0]
    if n_models == 1:
        return xs[0], x_stds[0]
    else:
        avg_x = list(huber_mean(xs=xs, **huber_kwargs))
        avg_xstd = list(huber_mean(xs=x_stds, **huber_kwargs))
        return avg_x, avg_xstd


if __name__=='__main__':
    xs = [[1,1,3],[0.1,-1,3]]
    x_std = [[0.1,4,5],[1,1,1]]
    print(combine_using_mean(xs=xs, x_stds=x_std))
    print(combine_using_median(xs=xs, x_stds=x_std))
    print(combine_using_huber(xs=xs, x_stds=x_std, a=1, b=1.5))

    xs = [[1, 1, 3]]
    x_std = [[0.1, 4, 5]]
    print(combine_using_mean(xs=xs, x_stds=x_std))
    print(combine_using_median(xs=xs, x_stds=x_std))
    print(combine_using_huber(xs=xs, x_stds=x_std, a=0.01, b=1.5))