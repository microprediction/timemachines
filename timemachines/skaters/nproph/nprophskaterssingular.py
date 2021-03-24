from timemachines.skaters.nproph.nprophskaterfactory import fbnprophet_skater_factory, fbnprophet_hyperparam_skater_factory
from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE, wrap
import numpy as np

# A collection of skaters powered by fbnprophet
# You'll want to read this review: https://www.microprediction.com/blog/nprophet


def fbnprophet_exogenous(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
    """ Uses known-in-advance and also y[1:] brought forward """
    return fbnprophet_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e)


def fbnprophet_recursive(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
    """ Same as exogenous, but uses nprophet to predict y[1:]  """
    return fbnprophet_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, recursive=True)


def fbnprophet_known(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
    """ Uses known-in-advance but not y[1:] """
    y0 = [wrap(y)[0]]
    return fbnprophet_skater_factory(y=y0, s=s, k=k, a=a, t=t, e=e)


def fbnprophet_univariate(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
    """ Simple univariate prediction using only y[0], and not 'a' or y[1:] """
    y0 = [wrap(y)[0]]
    return fbnprophet_skater_factory(y=y0, s=s, k=k, a=None, t=t, e=e)



def fbnprophet_cautious(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
    """ Similar to fbexogenous, but no crazy nonsense """
    if not s.get('s'):
        s['s']={}       # nprophet's state
        s['y']=list()   # maintain last five values
    y0 = wrap(y)[0]
    s['y'].append(y0)
    if len(s['y'])>5:
        s['y'].pop(0)
    import math
    x_upper = [ np.max(s['y'])+math.sqrt(j+1)*np.std(s['y']) for j in range(k) ]
    x_lower = [ np.min(s['y'])-math.sqrt(j+1)*np.std(s['y']) for j in range(k) ]
    x, x_std, s['s'] = fbnprophet_univariate(y=y,s=s['s'],k=k,a=a,t=t,e=e)
    x_careful = np.minimum(np.array(x),np.array(x_upper))
    x_careful = np.maximum(x_careful, np.array(x_lower))
    return list(x_careful), x_std, s


NPROPHET_SKATERS_SINGULAR = [fbnprophet_exogenous, fbnprophet_known, fbnprophet_univariate, fbnprophet_recursive,
                            fbnprophet_cautious]


# (1) Skaters with author-suggested two-dimensional hyper-parameter spaces

def fbnprophet_exogenous_r2(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r:R_TYPE=None):
    """ A skater using exogenous variables, with hyper-param tuning as recommended by authors """
    assert r is not None
    param_names = ['changepoint_prior_scale', 'seasonality_prior_scale']
    return fbnprophet_hyperparam_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, param_names=param_names)


def fbnprophet_recursive_r2(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r:R_TYPE=None):
    """ Same as exogenous, but uses nprophet to predict y[1:]  """
    assert r is not None
    param_names = ['changepoint_prior_scale', 'seasonality_prior_scale']
    return fbnprophet_hyperparam_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, param_names=param_names, recursive=True)


def fbnprophet_known_r2(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r:R_TYPE=None):
    """ Uses known-in-advance but not y[1:] """
    assert r is not None
    y0 = [wrap(y)[0]]
    param_names = ['changepoint_prior_scale', 'seasonality_prior_scale']
    return fbnprophet_hyperparam_skater_factory(y=y0, s=s, k=k, a=a, t=t, e=e, r=r, param_names=param_names, recursive=False)


def fbnprophet_univariate_r2(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None,r:R_TYPE=None):
    """ Simple univariate prediction using only y[0], and not 'a' or y[1:] """
    assert r is not None
    y0 = [wrap(y)[0]]
    param_names = ['changepoint_prior_scale', 'seasonality_prior_scale']
    return fbnprophet_hyperparam_skater_factory(y=y0, s=s, k=k, a=None, t=t, e=e, r=r, param_names=param_names, recursive=False)


NPROPHET_R2_SKATERS = [ fbnprophet_exogenous_r2, fbnprophet_known_r2, fbnprophet_univariate_r2, fbnprophet_recursive_r2 ]



if __name__ == '__main__':
    from timemachines.skatertools.data import hospital_with_exog
    from timemachines.skatertools.visualization.priorplot import prior_plot
    import matplotlib.pyplot as plt
    k = 1
    y, a = hospital_with_exog(k=k, n=450, offset=True)
    f = fbnprophet_exogenous
    err2 = prior_plot(f=f, k=k, y=y, n=450, n_plot=50)
    print(err2)
    plt.show()
    pass


