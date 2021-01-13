from typing import List
import numpy as np
import divinity as dv
from timemachines.conventions import to_int_log_space, Y_TYPE, K_TYPE
from timemachines.utilities import suppression
from timemachines.utilities.suppression import suppress_output
from timemachines.plotting import prior_plot

# I don't really recommend this guy. For one thing the logs are hard to suppress ;)


def divinity_seasonal(y:Y_TYPE, s, k:K_TYPE, a=None, t=None, e=None, r=0.5):
    """ A partial wrapping of the divinity library with notable limitations:

         - Fits every invocation (slow)
         - Ignores exogenous variables

        This guy's a tad noisy for my liking.
    """

    # Allow r to control buffer length and burn-in
    max_buffer_len, burnin = to_int_log_space(p=r, bounds=[(250, 5000), (120, 140)])
    assert max_buffer_len>100, 'divine library assumes 100 lags at least '

    if s is None:
        s = dict(buffer=[])

    if y is None:
        return None, s, None   # Ignore suggestion to fit offline

    if isinstance(y,List):
        y = y[0]         # Ignore exogenous variables

    # Update buffer
    s['buffer'].append(y)
    if len(s['buffer'])>max_buffer_len+1000:
        s['buffer'] = s['buffer'][-max_buffer_len:]

    # Fit and predict, if warm, or just last value
    if len(s['buffer']) < burnin:
        return y, s, None
    else:
        with suppress_output():  # this doesn't work :(
            model = dv.divinity(forecast_length=k,optimise_trend_season_features = True)
            model.fit(np.array(s['buffer']))
            x = model.predict()[k-1]
        return x, s, None


if __name__=='__main__':
    err = prior_plot(f=divinity_seasonal, k=1, n=150)
    pass

