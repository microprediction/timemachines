import pandas as pd
import sys
import logging
from typing import List, Tuple, Any
from timemachines.skatertools.utilities.conventions import wrap
from timemachines.skatertools.utilities.epochtime import infer_freq_from_epoch, is_valid_freq, epoch_to_naive_datetime, EPOCH
from timemachines.skatertools.utilities.suppression import no_stdout_stderr
from timemachines.skaters.proph.prophparams import PROPHET_MODEL, PROPHET_META
from timemachines.skatertools.utilities.wrangling import transpose

try: 
    from prophet import Prophet
    using_prophet=True
except ImportError:
    using_prophet=False

logging.disable(sys.maxsize)
logging.getLogger('fbprophet').setLevel(logging.ERROR)

if using_prophet:

    # Wraps the core prophet prediction capability into a plain vanilla prediction function
    # Unlike most skaters, this "integrated" prophet skater (iskater) isn't computed by calling the skater
    # successively - the opposite is true. We fit once on the entire history every data point, and that's that.
    # Thus this function, which optionally returns the forecast dataframe and the model, doesn't contain the
    # ancillary information that the skater carries (i.e. the prediction parade etc).


    def prophet_iskater_factory(y: [[float]], k: int, a: List = None, t: List = None, e=None, freq: str = None, n_max=1000,
                                recursive: bool = False, model_params: dict = None, return_forecast=True):
        """
        :param y:           A list of observations, each a vector.
        :param k:           Number of steps ahead to predict
        :param a:           Known in advance observations - should be k more of these than y's
        :param t:           Epoch times of observations y. If len(t)=len(y)+k the last k are interpreted as future times.
        :param freq:        'D', '5T' etc, see https://github.com/pandas-dev/pandas/blob/master/pandas/tseries/frequencies.py
        :param n_max:       Maximum number of observations to use, should you wish to prevent prophet from slowing down
        :param recursive    If True, exogenous variables y[1], y[2],... will be predicted forward in time
                                 (obviously this adds to computation time)
        :returns: x         k-vector of predictions
                  x_std     k-vector of standard deviations
                  forecast  full forecast dataframe, familiar to users of fbprophet
        """
        if a:
            assert len(a) == len(y) + k

        if isinstance(y[0], float):
            y = [wrap(yj) for yj in y]

        # Conversion of epoch times to UTC datetime
        # User must supply times, len(y) or len(y)+k, or a valid frequency str
        if t is None:
            if freq is None or not freq:
                freq = PROPHET_META['freq']  # Just assume away ...
            else:
                assert is_valid_freq(freq), 'Freq ' + str(freq) + ' is not a valid frequency'
            dt = pd.date_range(start=EPOCH, periods=len(y), freq=freq)  # UTC
        else:
            freq = infer_freq_from_epoch(t)
            dt = epoch_to_naive_datetime(t)

        if len(dt) == len(y) + k:
            ta = dt
            dt = dt[:len(y)]
        else:
            assert len(dt) == len(y), 'Time vector t should be len(y) or len(y)+k'
            ta = None

        # Truncate history so that prophet doesn't take forever to fit
        y_shorter = y[-n_max:]
        a_shorter = a[-(n_max + k):] if a is not None else []  # may be empty
        dt_shorter = dt[-n_max:]

        # Massage data into Prophet friendly dataframe with columns y, y1, ..., yk, a0,...aj
        y_cols = ['y' + str(i) if i > 0 else 'y' for i in range(len(y_shorter[-1]))]
        if a:
            a_cols = ['a' + str(i) for i in range(len(a_shorter[-1]))]
            data = [list(yi) + list(ai) for yi, ai in zip(y_shorter, a_shorter[:-k])]
            df = pd.DataFrame(columns=y_cols + a_cols, data=data)
        else:
            data = [list(yi) for yi in y_shorter]
            df = pd.DataFrame(columns=y_cols, data=data)
        df['ds'] = dt_shorter

        # Instantiate Prophet model, ensure defaults are what we think they are
        kwargs_used = dict([(k, v) for k, v in PROPHET_MODEL.items()])
        if model_params:
            kwargs_used.update(model_params)
        m = Prophet(**kwargs_used)

        # Add regressors
        for y_col in y_cols[1:]:
            m.add_regressor(name=y_col)
        if a:
            for a_col in a_cols:
                m.add_regressor(name=a_col)

        # Fit the model every invocation ... there isn't any other way
        with no_stdout_stderr():
            m.fit(df)

        # Make future dataframe, adding known-in-advance variables
        future = m.make_future_dataframe(periods=k, freq=freq)
        if a:
            for j, a_col in enumerate(a_cols):
                future[a_col] = [ai[j] for ai in a_shorter]  # Known in advance
        if ta is not None:
            future['ds'] = ta  # override with user supplied future times

        # Next, we wish to add contemporaneously observed variables
        #
        # This is somewhat problematic, for how should we bring exogenously observed variables forward?
        # The simplest answer is, don't use them - only supply 1-vector y observations
        # prophet implicitly assumes all exogenous are known, which is a pretty big shortcoming.
        #
        # However, if we are trying to support y[1:], ...
        #   - It seems consistent to use prophet to predict these forward,
        #   - It also seems likely that this will lead to over-fitting.
        # I'm open to ideas here. Perhaps perform some hackery could effect attenuation of the coefficients
        # assigned to y[1],... such as jiggling past observations. For now we use prophet on each
        # one individually, feeding them the known in advance 'a' variables.

        n_exog = len(y[0]) - 1
        if n_exog > 0:
            for j,y_col in enumerate(y_cols):
                if j>0:
                    yj = [yi[j] for yi in y_shorter]
                    if recursive:
                        yj_hat, yj_hat_std, yj_forecast, yj_m = prophet_iskater_factory(y=yj, k=k, a=a_shorter, freq=freq,
                                                                                        n_max=n_max, recursive=False)
                    else:
                        yj_hat = [yj[-1]] * k
                    future[y_col] = yj + list(yj_hat)

        # Call the prediction function
        forecast = m.predict(future)
        x = list(forecast['yhat'].values[-k:])  # Use m.plot(forecast) to take a peak

        # Interpret confidence level difference as scale to be returned. TODO: set alpha properly so this really is 1-std
        x_std = list([u - l for u, l in zip(forecast['yhat_upper'].values[-k:], forecast['yhat_lower'].values[-k:])])

        if return_forecast:
            return x, x_std, forecast, m
        else:
            return x, x_std


    # The rest of this module contains redundant special cases of the above, purely for testing.

    def prophet_fit_and_predict_simple(y: [float], k: int, freq: str = None, model_params: dict = None) -> Tuple[
        List, List, Any, Any]:
        """ Simpler wrapper for testing - univariate only """

        df = pd.DataFrame(columns=['y'], data=y)
        freq = freq or PROPHET_META['freq']
        df['ds'] = pd.date_range(start=EPOCH, periods=len(y), freq=freq)  # UTC
        kwargs_used = dict([(k, v) for k, v in PROPHET_MODEL.items()])
        if model_params:
            kwargs_used.update(model_params)
        m = Prophet(**kwargs_used)
        with no_stdout_stderr():
            m.fit(df)
        future = m.make_future_dataframe(periods=k, freq=freq)
        forecast = m.predict(future)
        x = forecast['yhat'].values[-k:]  # Use m.plot(forecast) to take a peak
        x_std = [u - l for u, l in zip(forecast['yhat_upper'].values, forecast['yhat_lower'].values)]
        return x, x_std, forecast, m


    def prophet_fit_and_predict_with_time(y: [float], k: int, t: [float], model_params: dict = None) -> Tuple[
        List, List, Any, Any]:
        """ Simpler wrapper for testing - univariate only w/ supplied times """

        df = pd.DataFrame(columns=['y'], data=y)
        df['ds'] = epoch_to_naive_datetime(t)
        kwargs_used = dict([(k, v) for k, v in PROPHET_MODEL.items()])
        if model_params:
            kwargs_used.update(model_params)
        m = Prophet(**kwargs_used)
        with no_stdout_stderr():
            m.fit(df)
        freq = infer_freq_from_epoch(t)
        future = m.make_future_dataframe(periods=k, freq=freq)
        forecast = m.predict(future)
        x = forecast['yhat'].values[-k:]  # Use m.plot(forecast) to take a peak
        x_std = [u - l for u, l in zip(forecast['yhat_upper'].values, forecast['yhat_lower'].values)]
        return x, x_std, forecast, m


    def prophet_fit_and_predict_with_time_and_advance_time(y: [float], k: int, t: [float], model_params: dict = None) -> \
    Tuple[List, List, Any, Any]:
        """ Simpler wrapper for testing - univariate only w/ supplied times and future times  """
        assert len(t) == len(y) + k
        df = pd.DataFrame(columns=['y'], data=y)
        dt = epoch_to_naive_datetime(t)
        df['ds'] = dt[:len(y)]
        kwargs_used = dict([(k, v) for k, v in PROPHET_MODEL.items()])
        if model_params:
            kwargs_used.update(model_params)
        m = Prophet(**kwargs_used)
        with no_stdout_stderr():
            m.fit(df)
        freq = infer_freq_from_epoch(t)
        future = m.make_future_dataframe(periods=k, freq=freq)
        future['ds'] = dt
        forecast = m.predict(future)
        x = forecast['yhat'].values[-k:]  # Use m.plot(forecast) to take a peak
        x_std = [u - l for u, l in zip(forecast['yhat_upper'].values, forecast['yhat_lower'].values)]
        return x, x_std, forecast, m


    def prophet_fit_and_predict_with_advance_vars(y: [float], k: int, t: [float], a: [[float]],
                                                  model_params: dict = None) -> Tuple[List, List, Any, Any]:
        """ Simpler wrapper for testing - univariate w/ advance vars w/ supplied times and future times  """
        assert len(t) == len(y) + k
        assert len(a) == len(y) + k
        assert isinstance(y[0], float)
        a_cols = ['a' + str(i) for i in range(len(a[0]))]
        df = pd.DataFrame(columns=a_cols, data=a[:-k])
        df['y'] = y
        dt = epoch_to_naive_datetime(t)
        df['ds'] = dt[:len(y)]
        kwargs_used = dict([(k, v) for k, v in PROPHET_MODEL.items()])
        if model_params:
            kwargs_used.update(model_params)
        m = Prophet(**kwargs_used)
        for a_col in a_cols:
            m.add_regressor(name=a_col)
        with no_stdout_stderr():
            m.fit(df)
        freq = infer_freq_from_epoch(t)
        future = m.make_future_dataframe(periods=k, freq=freq)
        future['ds'] = dt
        full_a_data = transpose(a)
        for a_col, a_vals in zip(a_cols, full_a_data):
            future[a_col] = a_vals
        forecast = m.predict(future)
        x = forecast['yhat'].values[-k:]  # Use m.plot(forecast) to take a peak
        x_std = [u - l for u, l in zip(forecast['yhat_upper'].values, forecast['yhat_lower'].values)]
        return x, x_std, forecast, m


    def prophet_fit_and_predict_with_exog_and_advance_vars(y: [[float]], k: int, t: [float], a: [[float]],
                                                           model_params: dict = None) -> Tuple[List, List, Any, Any]:
        """ Simpler wrapper for testing - univariate w/ advance vars w/ supplied times and future times  """
        assert len(t) == len(y) + k
        assert len(a) == len(y) + k
        assert isinstance(y[0], List)
        a_cols = ['a' + str(i) for i in range(len(a[0]))]
        df = pd.DataFrame(columns=a_cols, data=a[:-k])
        Y = transpose(y)
        df['y'] = Y[0]
        n_exog = len(y[0]) - 1
        y_cols = ['y'] + ['y' + str(i) for i in range(1, len(y[0]))]
        for i in range(1, n_exog + 1):
            df['y' + str(i)] = Y[i][:len(y)]
        dt = epoch_to_naive_datetime(t)
        df['ds'] = dt[:len(y)]
        kwargs_used = dict([(k, v) for k, v in PROPHET_MODEL.items()])
        if model_params:
            kwargs_used.update(model_params)
        m = Prophet(**kwargs_used)
        for a_col in a_cols:
            m.add_regressor(name=a_col)
        for y_col in y_cols[1:]:
            m.add_regressor(name=y_col)
        with no_stdout_stderr():
            m.fit(df)
        freq = infer_freq_from_epoch(t)
        future = m.make_future_dataframe(periods=k, freq=freq)
        future['ds'] = dt
        full_a_data = transpose(a)
        for a_col, a_vals in zip(a_cols, full_a_data):
            future[a_col] = a_vals
        for i in range(1, n_exog + 1):  # Just bring forward
            future['y' + str(i)] = Y[i] + [Y[i][-1]] * k
        forecast = m.predict(future)
        x = forecast['yhat'].values[-k:]  # Use m.plot(forecast) to take a peak
        x_std = [u - l for u, l in zip(forecast['yhat_upper'].values, forecast['yhat_lower'].values)]
        return x, x_std, forecast, m


    def prophet_fit_and_predict_with_exog_and_advance_vars_no_t(y: [[float]], k: int, freq: str, a: [[float]],
                                                                model_params: dict = None) -> Tuple[List, List, Any, Any]:
        """ Simpler wrapper for testing - univariate w/ advance vars w/ supplied times and future times  """
        assert len(a) == len(y) + k
        assert isinstance(y[0], List)
        a_cols = ['a' + str(i) for i in range(len(a[0]))]
        df = pd.DataFrame(columns=a_cols, data=a[:-k])
        Y = transpose(y)
        df['y'] = Y[0]
        n_exog = len(y[0]) - 1
        y_cols = ['y'] + ['y' + str(i) for i in range(1, len(y[0]))]
        for i in range(1, n_exog + 1):
            df['y' + str(i)] = Y[i][:len(y)]
        df['ds'] = pd.date_range(EPOCH, periods=len(y), freq=freq)
        kwargs_used = dict([(k, v) for k, v in PROPHET_MODEL.items()])
        if model_params:
            kwargs_used.update(model_params)
        m = Prophet(**kwargs_used)
        for a_col in a_cols:
            m.add_regressor(name=a_col)
        for y_col in y_cols[1:]:
            m.add_regressor(name=y_col)
        with no_stdout_stderr():
            m.fit(df)
        future = m.make_future_dataframe(periods=k, freq=freq)
        full_a_data = transpose(a)
        for a_col, a_vals in zip(a_cols, full_a_data):
            future[a_col] = a_vals
        for i in range(1, n_exog + 1):  # Just bring forward
            future['y' + str(i)] = Y[i] + [Y[i][-1]] * k
        forecast = m.predict(future)
        x = forecast['yhat'].values[-k:]  # Use m.plot(forecast) to take a peak
        x_std = [u - l for u, l in zip(forecast['yhat_upper'].values, forecast['yhat_lower'].values)]
        return x, x_std, forecast, m
