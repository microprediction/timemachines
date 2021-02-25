from datetime import datetime
from typing import Union, List
import pandas as pd
import pytz
from timemachines.skatertools.utilities.arrivals import approx_dt


EPOCH  = datetime.utcfromtimestamp(0)                     # Unix epoch start time as datetime
RECENT = datetime(2021,1,1)
RECENT_SECONDS = round((RECENT - EPOCH).total_seconds())  # Epoch seconds for a more recent date

FloatOrList = Union[float,List[float]]
DatetimeOrList = Union[datetime,List[datetime]]


def epoch_to_naive_datetime(t:FloatOrList)->DatetimeOrList:
    """ Naive conversion """
    if isinstance(t,float):
        return epoch_to_naive_datetime([t])[0]
    else:
        return [datetime.utcfromtimestamp(tj) for tj in t]


def naive_datetime_to_epoch(d:DatetimeOrList)->FloatOrList:
    if isinstance(d,float):
        return naive_datetime_to_epoch([d])[0]
    else:
        return [ dj.replace(tzinfo=pytz.UTC).timestamp() for dj in d]


def is_valid_freq(freq):
    try:
        dr = pd.date_range(start=EPOCH,periods=3,freq=freq)
        return True
    except ValueError:
        return False


def infer_freq_from_epoch(t:[float])->str:
    """ Infer a frequency string
    :param t: [ float ] epoch times
    :return:
    """
    # https://github.com/pandas-dev/pandas/blob/master/pandas/tseries/frequencies.py
    dt = epoch_to_naive_datetime(t)
    dti = pd.DatetimeIndex(dt)
    pd_freq = pd.infer_freq(dti)
    if pd_freq is None:
        dt = approx_dt(t)
        if dt is not None:
            return pd.infer_freq(epoch_to_naive_datetime([j * dt for j in range(20)]))
        else:
            return None
    else:
        return pd_freq
