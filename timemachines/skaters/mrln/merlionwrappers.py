from timemachines.skaters.mrln.merlioninclusion import using_merlion

if using_merlion:
    import numpy as np
    import pandas as pd
    from typing import List
    from merlion.utils.time_series import TimeSeries
    from merlion.models.forecast.prophet import Prophet, ProphetConfig
    from merlion.models.forecast.smoother import MSES, MSESConfig
    from merlion.models.forecast.arima import Arima, ArimaConfig

    def merlion_ARIMA_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, deseasonalize=False):
        """
            Calls Merlion's ARIMA forecasting model, but ignores t if supplied.
        """
        if a:
            assert len(a) == len(y) + k
        if np.isscalar(y[0]):
            y0s = [ yt for yt in y]
        else:
            y0s = [ yt[0] for yt in y ]

        y0s.extend([0]*k)

        train_val_list = ([1] * len(y)) + ([0] * k)
        idx = pd.date_range(end='01/01/2000', periods=len(y0s), freq="H")
        df = pd.DataFrame(y0s, index=idx).rename(columns={0:'y'})

        train_data = TimeSeries.from_pd(df.head(df.shape[0]-k))
        test_data = TimeSeries.from_pd(df.tail(k))

        model = Arima(ArimaConfig())
        model.train(train_data=train_data)
        test_pred, test_err = model.forecast(time_stamps=test_data.time_stamps)


        x = test_pred.to_pd()['y'].values.tolist()
        x_std = [1]*k

        return x, x_std

    if True:
        # TODO: FIXME: This ain''t working
        def merlion_MSES_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, deseasonalize=False):
            """
                Calls Merlion's MSES forecasting model, but ignores t if supplied.
            """
            if a:
                assert len(a) == len(y) + k
            if np.isscalar(y[0]):
                y0s = [ yt for yt in y]
            else:
                y0s = [ yt[0] for yt in y ]

            y0s.extend([0]*k)

            train_val_list = ([1] * len(y)) + ([0] * k)
            idx = pd.date_range(end='01/01/2000', periods=len(y0s), freq="H")
            df = pd.DataFrame(y0s, index=idx).rename(columns={0:'y'})

            train_data = TimeSeries.from_pd(df.head(df.shape[0]-k))
            test_data = TimeSeries.from_pd(df.tail(k))

            model = MSES(MSESConfig(k))
            model.train(train_data=train_data)
            test_pred, test_err = model.forecast(time_stamps=test_data.time_stamps)

            x = test_pred.to_pd()['y'].values.tolist()
            x_std = [1]*k

            return x, x_std

    if True:
        def merlion_Prophet_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, deseasonalize=False):
            """
                Calls Merlion's implemenetation of the Prophet forecasting model, but ignores t if supplied.
            """
            if a:
                assert len(a) == len(y) + k
            if np.isscalar(y[0]):
                y0s = [ yt for yt in y]
            else:
                y0s = [ yt[0] for yt in y ]

            y.extend([0]*k)

            train_val_list = ([1] * len(y)) + ([0] * k)
            idx = pd.date_range(end='01/01/2000', periods=len(y0s), freq="H")
            df = pd.DataFrame(y0s, index=idx).rename(columns={0:'y'})

            train_data = TimeSeries.from_pd(df.head(df.shape[0]-k))
            test_data = TimeSeries.from_pd(df.tail(k))

            model = Prophet(ProphetConfig())
            model.train(train_data=train_data)
            test_pred, test_err = model.forecast(time_stamps=test_data.time_stamps)

            x = test_pred.to_pd()['y'].values.tolist()
            x_std = [1]*k

            return x, x_std


if __name__=='__main__':
    assert using_merlion, 'pip install salesforce-merlion'
    import numpy as np
    import time
    st = time.time()
    y = list(np.cumsum(np.random.randn(400)))
    x, x_std = merlion_ARIMA_iskater(y=y, k=5)
    print(x)
    print(time.time()-st)