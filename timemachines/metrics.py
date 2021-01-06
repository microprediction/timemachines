
from timemachines.synthetic import brownian_with_noise, brownian_with_exogenous
from timemachines.iterative import prior
from sklearn.metrics import mean_squared_error



def brownian_rmse(f, ys=None, k=1, n=200):
    """ Useful for a quick test """
    if ys is None:
       ys = brownian_with_noise(n=n)
    xs = prior(f=f,ys=ys,k=1,ats=None, ts=None)
    rmse = mean_squared_error(ys[k:], xs[k:], squared=False)
    return rmse


def exogenous_rmse(f, ys=None, k=1, n=200):
    """ Useful for a quick test """
    if ys is None:
       ys = brownian_with_exogenous(n=n)
    xs = prior(f=f,ys=ys,k=1,ats=None, ts=None)
    y0 = [y_[0] for y_ in ys]
    rmse = mean_squared_error(y0[k:], xs[k:], squared=False)
    return rmse