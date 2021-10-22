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

    def pycrt_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, n_select=3):
        """
            Uses pycaret on univariate series (for now)
        """
        if a:
            assert len(a) == len(y) + k
        if np.isscalar(y[0]):
            y0s = [ yt for yt in y]
        else:
            y0s = [ yt[0] for yt in y ]

        y0_frame = pd.Series(index=pd.PeriodIndex(pd.date_range("2021-01", periods=len(y0s), freq="H")), data=y0s).to_frame()
        last_t = y0_frame.index[-1]
        next_t = last_t.to_timestamp() + datetime.timedelta(hours=1)
        fold = 3
        with no_stdout_stderr():
            # Full fit cycle each time, for now
            exp = TimeSeriesExperiment()
            exp.setup(data=y0_frame)
            best_baseline_models = exp.compare_models(fold=fold, sort='smape', n_select=n_select)
            fh = ForecastingHorizon(pd.PeriodIndex(pd.date_range(next_t, periods=k, freq="H")), is_relative=False)
            X = np.ndarray(shape=(n_select,k))
            for ndx,model in enumerate(best_baseline_models):
                  x_ = model.predict(fh)
                  X[ndx][:] = x_
        x = list(np.nanmedian(X,axis=0))
        x_std = list(np.nanstd(X,axis=0))
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