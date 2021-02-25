from timemachines.skatertools.utilities.conventions import positive_log_scale, to_space

# Some "global" prophet defaults set outside the model

PROPHET_META={'n_warm':400,  # Default minimum number of data points before fitting
              'freq':'H',    # Sampling frequency used if none is supplied, or t not supplied
              'n_max':500}   # Maximum length of training data

# Some prophet model parameters

PROPHET_MODEL = dict(changepoint_prior_scale=0.05,  # Suggested  [0.001, 0.01, 0.1, 0.5],
                    seasonality_prior_scale = 10.0,  # Suggested  [0.01, 0.1, 1.0, 10.0],
                    holidays_prior_scale = 10.0,  # Suggested  [0.01, 0.1, 1.0, 10.0]
                    seasonality_mode = 'additive',  # ['additive','multiplicative']
                    changepoint_range = 0.8,  # Suggested  [0.8-0.95]
                    interval_width = 0.34134*2)   # So it is +/- one standard deviation

PROPHET_MODEL_LOG_LOW = dict(changepoint_prior_scale=0.001,  # Suggested  [0.001, 0.01, 0.1, 0.5],
                    seasonality_prior_scale = 0.01,  # Suggested  [0.01, 0.1, 1.0, 10.0],
                    holidays_prior_scale = 0.01)  # Suggested  [0.01, 0.1, 1.0, 10.0]


PROPHET_MODEL_LOG_HIGH = dict(changepoint_prior_scale=0.5,  # Suggested  [0.001, 0.01, 0.1, 0.5],
                    seasonality_prior_scale = 20.,  # Suggested  [0.01, 0.1, 1.0, 10.0],
                    holidays_prior_scale = 20.,  # Suggested  [0.01, 0.1, 1.0, 10.0]
                    changepoint_range = 0.95)  # Suggested  [0.8-0.95]

PROPHET_MODEL_LINEAR_LOW  = dict(changepoint_range = 0.8)  # Suggested  [0.8-0.95]
PROPHET_MODEL_LINEAR_HIGH = dict(changepoint_range = 0.8)  # Suggested  [0.8-0.95]


def prophet_params(r:float,dim:int, param_names:[str])->dict:
    """ Interpret r in (0,1) as dict of param values """
    u = to_space(r, dim=dim)
    return dict([(name, prophet_param(name, ui)) for name, ui in zip(param_names, u)])


def prophet_param(param_name, u:float):
    """ Map u in (0,1) to sensible parameter value """
    assert 0 <= u <= 1
    if param_name in PROPHET_MODEL_LINEAR_LOW:
        lb = PROPHET_MODEL_LINEAR_LOW[param_name]
        ub = PROPHET_MODEL_LINEAR_HIGH[param_name]
        return (ub-lb) * u + lb
    elif param_name in PROPHET_MODEL_LOG_LOW:
        lb = PROPHET_MODEL_LOG_LOW[param_name]
        ub = PROPHET_MODEL_LOG_HIGH[param_name]
        return positive_log_scale(u=u,low=lb,high=ub)
    else:
        raise ValueError()



class ProphetWarning(UserWarning):
    pass