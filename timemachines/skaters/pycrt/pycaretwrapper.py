from timemachines.skaters.pycrt.pycaretinclusion import using_pycaret
from timemachines.inclusion.pandasinclusion import using_pandas
from timemachines.skaters.sk.skinclusion import using_sktime
import datetime
from typing import List


if using_pycaret and using_pandas and using_sktime:
    import pandas as pd
    import numpy as np
    from pycaret.internal.pycaret_experiment import TimeSeriesExperiment
    from sktime.forecasting.base import ForecastingHorizon
    from timemachines.skatertools.utilities.suppression import no_stdout_stderr

    def pycrt_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, n_select=3, blend_method='median'):
        """
            Uses pycaret on univariate series (for now)
        """
        # Examples: https://github.com/pycaret/pycaret/blob/time_series_beta/time_series_101.ipynb
        if a:
            assert len(a) == len(y) + k
        if np.isscalar(y[0]):
            y0s = [ yt for yt in y]
        else:
            y0s = [ yt[0] for yt in y ]

        y0_frame = pd.Series(index=pd.PeriodIndex(pd.date_range("2021-01", periods=len(y0s), freq="H")), data=y0s).to_frame()
        #last_t = y0_frame.index[-1]
        #next_t = last_t.to_timestamp() + datetime.timedelta(hours=1)
        fold = 3
        with no_stdout_stderr():
            # Full fit cycle each time, for now
            exp = TimeSeriesExperiment()
            exp.setup(data=y0_frame, n_jobs=1, fh=k)
            best_baseline_models = exp.compare_models(fold=fold, sort='smape', n_select=n_select,exclude=["auto_arima"])
            best_tuned_models = [exp.tune_model(model) for model in best_baseline_models]
            blended_model = exp.blend_models(best_tuned_models, method=blend_method)
            x = list(exp.predict_model(blended_model))
            # fh = ForecastingHorizon(pd.PeriodIndex(pd.date_range(next_t, periods=k, freq="H")), is_relative=False)
            #X = np.ndarray(shape=(n_select,k))
            #for ndx,tuned_model in enumerate(best_tuned_models):
            #      x_ = tuned_model.predict(fh)
            #      X[ndx][:] = x_
        x_std = [ 1 for _ in x ] # Will be overridden by empirical
        return x, x_std

if __name__=='__main__':
    assert using_pycaret, 'pip install pycaret-ts-alpha'
    import numpy as np
    import time
    st = time.time()
    y = list(np.cumsum(np.random.randn(400)))
    x, x_std = pycrt_iskater(y=y, k=5)
    print('x')
    print(x)
    print('x_std')
    print(x_std)
    print('Elapsed')
    print(time.time()-st)