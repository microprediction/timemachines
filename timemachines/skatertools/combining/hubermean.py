# Ack: https://arxiv.org/pdf/2108.12627.pdf

import numpy as np
import math
from scipy.optimize import newton

# TODO: Add precision weighting etc

# This version is more up to date than the version used in precise.
# TODO: Move Huber into separate little package, groan.


def huber_mean(xs:[[float]], a:float=1.0, b=2.0, n_iter=20, atol=1e-8)->[float]:
    """ Compute a columnwise pseudo-mean of xs, by minimizing a generalized Huber error that is
        proportional to x^2 near zero and asymptotes to |x| as |x|->infinity.
               f(x) = 1/a log( exp(a*(x-mu)) + exp(-(a*(x-mu)) + b )
        This is the same as the function below, except the parameter a will multiply std(x)
        :param xs:    (n_samples, n_vars)
        :param a:    Generalized Huber parameter as per formula
        :param b:    Generalized Huber parameter as per formula above, scalar or (nvars,)
        :param with_fraction_converged: If True, will return the mean number of rows where convergence was achieved
        :return:      (n_vars,)   location parameters

    """
    if len(xs)==0:
        raise ValueError('Cannot compute huber mean for no data')
    elif len(xs)==1:
        return xs[0]
    else:
        x_std = np.nanstd(xs,axis=0)
        xs_rescaled = xs/(10*x_std+0.0001)
        a_abs = a/10.
        scaled_mean = huber_mean_absolute_params(xs=xs_rescaled, a=a_abs, b=b, n_iter=n_iter, atol=atol, with_fraction_converged=False)
        unscaled_mean = scaled_mean * x_std*10
        return unscaled_mean


def huber_mean_absolute_params(xs:[[float]], a, b, n_iter=20, atol=1e-8, with_fraction_converged=False)->[float]:
    """ Finds the generalized Huber locations for many variables at once
        Each column of xs represents a different variable whose pseudo-mean will be computed
        Thus the result mu might be compared to np.mean(xs, axis=0)

        The function being minimized w.r.t mu is
              f(x) = 1/a log( exp(a*(x-mu)) + exp(-(a*(x-mu)) + b )
        and for |x|->0 this has asymptote:
             f(x) ->  log(2+b)/a + a/(2+b) * (x-mu)^2
        whereas for |x|->infinity
             f(x) -> |x-mu|

        This Huber function is not the standard Huber loss https://en.wikipedia.org/wiki/Huber_loss
        Rather, it is based on https://arxiv.org/pdf/2108.12627.pdf

    :param xs:    (n_samples, n_vars)
    :param a:    Generalized Huber parameter as per formula above, scalar or (nvars,)
    :param b:    Generalized Huber parameter as per formula above, scalar or (nvars,)
    :return:      (n_vars,)   location parameters
    """
    x_median = np.median(xs, axis=0)
    x_mean = np.mean(xs, axis=0)
    lb = np.where(x_median < x_mean, x_median, x_mean)-0.01
    ub = np.where(x_median > x_mean, x_median, x_mean)+0.01
    x0 = 0.75*lb+0.25*ub
    flb = huber_deriv(lb,a,b,xs)
    fub = huber_deriv(ub,a,b,xs)
    mu = newton(func=huber_deriv, x0=x0, args=(a, b, xs), tol=1e-6, rtol=1e-4 )
    return mu


def huber_deriv(mu, a, b, xs):
    """ Derivative of generalized Huber loss w.r.t. mu

         f'(x) = 1/a log( exp(a*(x-mu)) + exp(-(a*(x-mu)) + b )

    :param mu:   (n_samples,)                    # Vector of location parameters
    :param xs :   (n_samples,n_vars)              # Data
    :param a:    Generalized Huber parameter(s)
    :param b:    Generalized Huber parameter(s)
    :return:
    """
    n_samples, n_vars = np.shape(xs)
    mu_rep = np.tile(np.atleast_2d(mu), (n_samples,1))
    y = xs - mu_rep
    chain_rule = -1.0
    numer = np.exp(np.dot(y,a) ) - np.exp(-np.dot(a,y))  # (n_samples, n_params)
    denom = numer + b
    gradient = chain_rule*numer/denom
    return np.mean(gradient,axis=0)


def huber_abs_error(mu, a, b, xs):
    """ Generalized Huber loss which is "like" abs error, as it approaches |x-mu| as |x-mu|-> infinity
          f(x) = 1/a log( exp(a*(x-mu)) + exp(-(a*(x-mu)) + b )
     """
    n_samples, n_vars = np.shape(xs)
    mu_rep = np.tile(np.atleast_2d(mu), (n_samples, 1))
    y = xs - mu_rep
    numer = np.log(np.exp(np.dot(y, a)) + np.exp(np.dot(a, y)) + b)  # (n_samples, n_params)
    denom = a
    return numer / denom


def mean_huber_linear_error(mu, a, b, xs):
    """
        This convenience function returns columnwise mean.
    """
    f_val = huber_abs_error(mu=mu, a=a, b=b, xs=xs)
    return np.mean(f_val, axis=0)


def huber_squared_error(mu, a, b, xs):
    """ Rescaled generalized Huber loss which is "like" squared error, in
    the sense that it approaches (x-mu)^2 as |x-mu|-> 0

        If
         f(x) = 1/a log( exp(a*(x-mu)) + exp(-(a*(x-mu)) + b )
        Then
         f(x) -> log(2+b)/a + a/(2+b) * x^2   as x->0
        by Taylor. So we define
         g(x) :=  ( f(x) - log(2+b)/a ) * (2+b)/a

    """
    f = huber_abs_error(mu=mu,a=a,b=b,xs=xs)
    c = math.log(2 + b) / a
    d = a / (2 + b)
    g = ( f - c ) / d
    return g


def mean_huber_squared_error(mu, a, b, xs):
    """
        This convenience function returns columnwise mean.
    """
    f_val = huber_squared_error(mu=mu, a=a, b=b, xs=xs)
    return np.mean(f_val, axis=0)


def mean_quadratic_error(mu, xs):
    """
       Here as a useful comparison to the below
    """
    n_samples, n_vars = np.shape(xs)
    mu_rep = np.tile(np.atleast_2d(mu), (n_samples, 1))
    y = xs - mu_rep
    return np.nanmean(y**2, axis=0)
