from timemachines.skaters.kts.ktsinclusion import using_kats

if using_kats:
    import numpy as np
    import pandas as pd
    from typing import List
    from kats.consts import TimeSeriesData
    from kats.models.prophet import ProphetModel, ProphetParams
    from kats.models.holtwinters import HoltWintersModel, HoltWintersParams
    from kats.models.quadratic_model import QuadraticModel, QuadraticModelParams


    def kats_Prophet_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, deseasonalize=False):
        """
            Calls Kats' Prophet forecasting model, but ignores t if supplied.
        """
        if a:
            assert len(a) == len(y) + k
        if np.isscalar(y[0]):
            y0s = [ yt for yt in y]
        else:
            y0s = [ yt[0] for yt in y ]

        idx = pd.date_range(end='01/01/2000', periods=len(y0s), freq="H")
        df = pd.DataFrame(y0s, index=idx).rename(columns={0:'y'}).reset_index()
        df.columns = ["time", "value"]

        train_data = TimeSeriesData(df.head(df.shape[0]))

        params = ProphetParams(seasonality_mode='multiplicative') # additive mode gives worse results
        m = ProphetModel(train_data, params)
        m.fit()
        fcst = m.predict(steps=k, freq="H")

        x = list(fcst.fcst.values)
        x_std = [1]*k

        return x, x_std

    if True:
        def kats_HoltWinters_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, deseasonalize=False):
            """
                Calls Kats' Holt-Winters forecasting model, but ignores t if supplied.
            """
            if a:
                assert len(a) == len(y) + k
            if np.isscalar(y[0]):
                y0s = [ yt for yt in y]
            else:
                y0s = [ yt[0] for yt in y ]

            idx = pd.date_range(end='01/01/2000', periods=len(y0s), freq="H")
            df = pd.DataFrame(y0s, index=idx).rename(columns={0:'y'}).reset_index()
            df.columns = ["time", "value"]

            train_data = TimeSeriesData(df.head(df.shape[0]))

            params = HoltWintersParams(
                        trend="add"
                        #damped=False,
                        #seasonal="mul",
                        #seasonal_periods=12,
                    )

            m = HoltWintersModel(
                data=train_data, 
                params=params
                )

            m.fit()
            fcst = m.predict(steps=k, alpha = 0.5)

            x = list(fcst.fcst.values)
            x_std = [1]*k

            return x, x_std

    if True:
        def kats_quadratic_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, deseasonalize=False):
            """
                Calls Kats' quadratic regressions forecasting model, but ignores t if supplied.
            """
            if a:
                assert len(a) == len(y) + k
            if np.isscalar(y[0]):
                y0s = [ yt for yt in y]
            else:
                y0s = [ yt[0] for yt in y ]

            idx = pd.date_range(end='01/01/2000', periods=len(y0s), freq="H")
            df = pd.DataFrame(y0s, index=idx).rename(columns={0:'y'}).reset_index()
            df.columns = ["time", "value"]

            train_data = TimeSeriesData(df.head(df.shape[0]))

            params = QuadraticModelParams()

            m = QuadraticModel(train_data, params)
            m.fit()
            fcst = m.predict(steps = k)

            x = list(fcst.fcst.values)
            x_std = [1]*k

            return x, x_std


if __name__=='__main__':
    assert using_kats, 'pip install kats'
    import numpy as np
    import time
    st = time.time()
    y = list(np.cumsum(np.random.randn(400)))
    x, x_std = kats_Prophet_iskater(y=y, k=5)
    print(x)
    print(time.time()-st)