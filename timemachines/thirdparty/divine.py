from typing import List
import numpy as np
import divinity as dv
import math
from timemachines.conventions import to_parameters_logscale, Y_TYPE, K_TYPE, rmse1, prior_plot
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',FutureWarning)


def divine(y:Y_TYPE, s, k:K_TYPE, a=None, t=None, e=None, r=0.5):
    """ A simple wrapping of the divinity library
         - Fitting each call (slow)
         - Ignores exogenous variables
    """

    # Allow r to control buffer length and burn-in
    prms = to_parameters_logscale(r=r, exponents=[6, 2], bounds=[(10, 20), (125, 250)])
    burnin = max( int(math.ceil(prms[1])), 100)   # library limitation with default params
    max_buffer_len = int(math.ceil(prms[0]))

    if s is None:
        s = dict(buffer=[])

    if y is None:
        return None, s   # Ignore suggestion to fit offline

    if isinstance(y,List):
        y = y[0]         # Ignore exogenous variables

    # Update buffer
    s['buffer'].append(y)
    if len(s['buffer'])>max_buffer_len+1000:
        s['buffer'] = s['buffer'][-max_buffer_len:]

    # Fit and predict, if warm, or just last value
    if len(s['buffer']) < burnin:
        return y, s
    else:
        model = dv.divinity(forecast_length=k)
        model.fit(np.array(s['buffer']))
        x = model.predict()[k-1]
        return x, s


if __name__=='__main__':
    err = prior_plot(f=divine, k=1, n=150)
    pass

