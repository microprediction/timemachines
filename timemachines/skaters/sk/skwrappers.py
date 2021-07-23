
from timemachines.skaters.sk.skinclusion import using_sktime

if using_sktime:
    import numpy as np
    import pandas as pd
    from sktime.forecasting.theta import ThetaForecaster
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.arima import AutoARIMA
    import datetime
    from typing import List
    from timemachines.skatertools.utilities.suppression import no_stdout_stderr

    def sk_theta_hourly_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, deseasonalize=False):
        """
            Calls sktime's theta forecaster implementation, but ignores t if supplied.
        """
        if a:
            assert len(a) == len(y) + k
        if np.isscalar(y[0]):
            y0s = [ yt for yt in y]
        else:
            y0s = [ yt[0] for yt in y ]

        y0_series = pd.Series(index=pd.PeriodIndex(pd.date_range("2021-01", periods=len(y0s), freq="H")), data=y0s)
        last_t = y0_series.index[-1]
        next_t = last_t.to_timestamp() + datetime.timedelta(hours=1)
        with no_stdout_stderr():
            forecaster = ThetaForecaster(deseasonalize=deseasonalize)
            forecaster.fit(y0_series)
            fh = ForecastingHorizon(pd.PeriodIndex(pd.date_range(next_t, periods=k, freq="H")), is_relative=False )
            x = forecaster.predict(fh)
        x_std = [1.0 for _ in x]
        return x, x_std

    def sk_autoarima_hourly_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None,
                                    start_p=2, d=None, start_q=2, max_p=5, max_d=2, max_q=5, start_P=1, D=None, start_Q=1,
                                    max_P=2, max_D=1, max_Q=2, max_order=5, sp=1, seasonal=True, stationary=False,
                                    information_criterion='aic', alpha=0.05, test='kpss', seasonal_test='ocsb', stepwise=True,
                                    n_jobs=1, start_params=None, trend=None, method='lbfgs', maxiter=50, offset_test_args=None,
                                    seasonal_test_args=None, suppress_warnings=True, n_fits=10, out_of_sample_size=0, scoring='mse',
                                    scoring_args=None, with_intercept=True, **kwargs):
        """
           TODO: Fix if t is supplied, similar to with prophet
        """
        if a:
            assert len(a) == len(y) + k
        if np.isscalar(y[0]):
            y0s = [yt for yt in y]
        else:
            y0s = [yt[0] for yt in y]
        y0_series = pd.Series(index=pd.PeriodIndex(pd.date_range("2021-01", periods=len(y0s), freq="H")), data=y0s)
        last_t = y0_series.index[-1]
        next_t = last_t.to_timestamp() + datetime.timedelta(hours=1)
        with no_stdout_stderr():
            forecaster = AutoARIMA(start_p=start_p, d=d, start_q=start_q, max_p=max_p, max_d=max_d, max_q=max_q,
                                   start_P=start_P, D=None, start_Q=start_Q,
                             max_P=max_P, max_D=max_D, max_Q=max_Q, max_order=max_order, sp=sp, seasonal=seasonal, stationary=stationary,
                             information_criterion=information_criterion, alpha=alpha, test=test, seasonal_test=seasonal_test, stepwise=stepwise,
                             n_jobs=n_jobs, start_params=start_params, trend=trend, method=method, maxiter=maxiter, offset_test_args=offset_test_args,
                             seasonal_test_args=seasonal_test_args, suppress_warnings=suppress_warnings, error_action='warn', trace=False,
                             n_fits=n_fits, out_of_sample_size=out_of_sample_size, scoring=scoring,
                             scoring_args=scoring_args, with_intercept=with_intercept)
            # For argument meanings see: https://www.sktime.org/en/latest/api_reference/modules/auto_generated/sktime.forecasting.arima.AutoARIMA.html
            forecaster.fit(y0_series)
            fh = ForecastingHorizon(pd.PeriodIndex(pd.date_range(next_t, periods=k, freq="H")), is_relative=False)
            x = forecaster.predict(fh)
        x_std = [1.0 for _ in x]
        return x, x_std



if __name__=='__main__':
    assert using_sktime, 'pip install sktime'
    import numpy as np
    import time
    st = time.time()
    y = list(np.cumsum(np.random.randn(400)))
    x, x_std = sk_autoarima_hourly_iskater(y=y, k=5)
    print(x)
    print(time.time()-st)
