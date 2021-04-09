from timemachines.skatertools.utilities.conventions import positive_log_scale, to_space

# Some "global" nprophet defaults set outside the model

NPROPHET_META={
    'n_warm':400,  # Default minimum number of data points before fitting
    'freq':'H',    # Sampling frequency used if none is supplied, or t not supplied
    'n_max':500    # Maximum length of training data
}   

# Some nprophet model parameters

NPROPHET_MODEL = {
    'growth': 'linear',
    'changepoints': None,
    'n_changepoints': 10,
    'changepoints_range': 0.9,
    'trend_reg': 0,
    'trend_reg_threshold': False,
    'yearly_seasonality': 'auto',
    'weekly_seasonality': 'auto',
    'daily_seasonality': 'auto',
    'seasonality_mode': 'additive',
    'seasonality_reg': 0,
    'n_forecasts': 1,
    'n_lags': 3,
    'num_hidden_layers': 0,
    'd_hidden': None,
    'ar_sparsity': None,
    'learning_rate': None,
    'epochs': None,
    'batch_size': None,
    'loss_func': 'Huber',
    'optimizer': 'AdamW',
    'train_speed': None,
    'normalize': 'auto',
    'impute_missing': True 
}

NPROPHET_MODEL_LOG_LOW = dict()
NPROPHET_MODEL_LOG_HIGH = dict()  
NPROPHET_MODEL_LINEAR_LOW  = dict()
NPROPHET_MODEL_LINEAR_HIGH = dict()


def nprophet_params(r:float,dim:int, param_names:[str])->dict:
    """ Interpret r in (0,1) as dict of param values """
    u = to_space(r, dim=dim)
    return dict([(name, nprophet_param(name, ui)) for name, ui in zip(param_names, u)])


def nprophet_param(param_name, u:float):
    """ Map u in (0,1) to sensible parameter value """
    assert 0 <= u <= 1
    if param_name in NPROPHET_MODEL_LINEAR_LOW:
        lb = NPROPHET_MODEL_LINEAR_LOW[param_name]
        ub = NPROPHET_MODEL_LINEAR_HIGH[param_name]
        return (ub-lb) * u + lb
    elif param_name in NPROPHET_MODEL_LOG_LOW:
        lb = NPROPHET_MODEL_LOG_LOW[param_name]
        ub = NPROPHET_MODEL_LOG_HIGH[param_name]
        return positive_log_scale(u=u,low=lb,high=ub)
    else:
        raise ValueError()



class ProphetWarning(UserWarning):
    pass