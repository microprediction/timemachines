from timemachines.skaters.proph.prophskaterfactory import fbprophet_skater_factory, fbprophet_hyperparam_skater_factory
from timemachines.skaters.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE, wrap


# A collection of skaters powered by fbprophet



# (1) Skaters with no hyper-parameters, intended for easy use

def fbprophet_exogenous(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
    """ Uses known-in-advance and also y[1:] brought forward """
    return fbprophet_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e)


def fbprophet_recursive(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
    """ Same as exogenous, but uses prophet to predict y[1:]  """
    return fbprophet_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, recursive=True)


def fbprophet_known(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
    """ Uses known-in-advance but not y[1:] """
    y0 = [wrap(y)[0]]
    return fbprophet_skater_factory(y=y0, s=s, k=k, a=a, t=t, e=e)


def fbprophet_univariate(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
    """ Simple univariate prediction using only y[0], and not 'a' or y[1:] """
    y0 = [wrap(y)[0]]
    return fbprophet_skater_factory(y=y0, s=s, k=k, a=None, t=t, e=e)

PROPHET_SKATERS = [ fbprophet_exogenous, fbprophet_known, fbprophet_univariate, fbprophet_recursive ]


# (1) Skaters with author-suggested two-dimensional hyper-parameter spaces

def fbprophet_exogenous_r2(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r:R_TYPE=None):
    """ A skater using exogenous variables, with hyper-param tuning as recommended by authors """
    assert r is not None
    param_names = ['changepoint_prior_scale', 'seasonality_prior_scale']
    return fbprophet_hyperparam_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, param_names=param_names)


def fbprophet_recursive_r2(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r:R_TYPE=None):
    """ Same as exogenous, but uses prophet to predict y[1:]  """
    assert r is not None
    param_names = ['changepoint_prior_scale', 'seasonality_prior_scale']
    return fbprophet_hyperparam_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, param_names=param_names, recursive=True)


def fbprophet_known_r2(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r:R_TYPE=None):
    """ Uses known-in-advance but not y[1:] """
    assert r is not None
    y0 = [wrap(y)[0]]
    param_names = ['changepoint_prior_scale', 'seasonality_prior_scale']
    return fbprophet_hyperparam_skater_factory(y=y0, s=s, k=k, a=a, t=t, e=e, r=r, param_names=param_names, recursive=False)


def fbprophet_univariate_r2(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None,r:R_TYPE=None):
    """ Simple univariate prediction using only y[0], and not 'a' or y[1:] """
    assert r is not None
    y0 = [wrap(y)[0]]
    param_names = ['changepoint_prior_scale', 'seasonality_prior_scale']
    return fbprophet_hyperparam_skater_factory(y=y0, s=s, k=k, a=None, t=t, e=e, r=r, param_names=param_names, recursive=False)


PROPHET_R2_SKATERS = [ fbprophet_exogenous_r2, fbprophet_known_r2, fbprophet_univariate_r2, fbprophet_recursive_r2 ]