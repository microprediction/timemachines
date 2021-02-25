from statistics import mode, StatisticsError
import numpy as np


def approx_dt(t:[float], default_dt=60 ):
    """ Crude estimate of the typical time between arrivals
          t: list of epoch times
    """
    if len(t) > 5:
        return approx_mode([abs(dt) for dt in np.diff(list(t))]) or default_dt
    else:
        return default_dt


def approx_mode(xs, ndigits=0):
    """ Mode of rounded numbers, or None """
    xr = [round(x, ndigits) for x in xs]
    try:
        return mode(xr)
    except StatisticsError:
        return None