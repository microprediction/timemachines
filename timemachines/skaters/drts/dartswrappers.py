from timemachines.skaters.drts.dartsinclusion import using_darts

if using_darts:
    import numpy as np
    import pandas as pd
    from typing import List
    from darts import TimeSeries
    from darts.models import FFT,Prophet,ARIMA,AutoARIMA,ExponentialSmoothing,Theta,FourTheta,TransformerModel,NBEATSModel
    from darts.utils.utils import SeasonalityMode

    def darts_FFT_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, deseasonalize=False):
        """
            Calls Darts' FFT forecasting model, but ignores t if supplied.
        """
        if a:
            assert len(a) == len(y) + k
        if np.isscalar(y[0]):
            y0s = [ yt for yt in y]
        else:
            y0s = [ yt[0] for yt in y ]

        y0s.extend([0]*k)
        idx = pd.date_range(end='01/01/2000', periods=len(y0s), freq="H")
        df = pd.DataFrame(y0s, index=idx).reset_index().rename(columns={'index':'ts',0:'y'})
        df.head()

        series = TimeSeries.from_dataframe(df, 'ts', 'y')
        train, val = series[:-k], series[-k:]

        model = FFT()
        model.fit(train)
        prediction = model.predict(len(val))

        x = prediction.pd_dataframe()['y'].values.tolist()
        x_std = [1]*k

        return x, x_std
        
    def darts_Prophet_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, deseasonalize=False):
        """
            Calls Darts' Prophet forecasting model, but ignores t if supplied.
        """
        if a:
            assert len(a) == len(y) + k
        if np.isscalar(y[0]):
            y0s = [ yt for yt in y]
        else:
            y0s = [ yt[0] for yt in y ]

        y0s.extend([0]*k)
        idx = pd.date_range(end='01/01/2000', periods=len(y0s), freq="H")
        df = pd.DataFrame(y0s, index=idx).reset_index().rename(columns={'index':'ts',0:'y'})
        df.head()

        series = TimeSeries.from_dataframe(df, 'ts', 'y')
        train, val = series[:-k], series[-k:]

        model = Prophet()
        model.fit(train)
        prediction = model.predict(len(val))

        x = prediction.pd_dataframe()['y'].values.tolist()
        x_std = [1]*k

        return x, x_std
        
    def darts_ARIMA_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, deseasonalize=False):
        """
            Calls Darts' ARIMA forecasting model, but ignores t if supplied.
        """
        if a:
            assert len(a) == len(y) + k
        if np.isscalar(y[0]):
            y0s = [ yt for yt in y]
        else:
            y0s = [ yt[0] for yt in y ]

        y0s.extend([0]*k)
        idx = pd.date_range(end='01/01/2000', periods=len(y0s), freq="H")
        df = pd.DataFrame(y0s, index=idx).reset_index().rename(columns={'index':'ts',0:'y'})
        df.head()

        series = TimeSeries.from_dataframe(df, 'ts', 'y')
        train, val = series[:-k], series[-k:]

        model = ARIMA()
        model.fit(train)
        prediction = model.predict(len(val))

        x = prediction.pd_dataframe()['y'].values.tolist()
        x_std = [1]*k

        return x, x_std
        
    def darts_AutoARIMA_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, deseasonalize=False):
        """
            Calls Darts' AutoARIMA forecasting model, but ignores t if supplied.
        """
        if a:
            assert len(a) == len(y) + k
        if np.isscalar(y[0]):
            y0s = [ yt for yt in y]
        else:
            y0s = [ yt[0] for yt in y ]

        y0s.extend([0]*k)
        idx = pd.date_range(end='01/01/2000', periods=len(y0s), freq="H")
        df = pd.DataFrame(y0s, index=idx).reset_index().rename(columns={'index':'ts',0:'y'})
        df.head()

        series = TimeSeries.from_dataframe(df, 'ts', 'y')
        train, val = series[:-k], series[-k:]

        model = AutoARIMA()
        model.fit(train)
        prediction = model.predict(len(val))

        x = prediction.pd_dataframe()['y'].values.tolist()
        x_std = [1]*k

        return x, x_std
        
    def darts_ExponentialSmoothing_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, deseasonalize=False):
        """
            Calls Darts' ExponentialSmoothing forecasting model, but ignores t if supplied.
        """
        if a:
            assert len(a) == len(y) + k
        if np.isscalar(y[0]):
            y0s = [ yt for yt in y]
        else:
            y0s = [ yt[0] for yt in y ]

        y0s.extend([0]*k)
        idx = pd.date_range(end='01/01/2000', periods=len(y0s), freq="H")
        df = pd.DataFrame(y0s, index=idx).reset_index().rename(columns={'index':'ts',0:'y'})
        df.head()

        series = TimeSeries.from_dataframe(df, 'ts', 'y')
        train, val = series[:-k], series[-k:]

        model = ExponentialSmoothing()
        model.fit(train)
        prediction = model.predict(len(val))

        x = prediction.pd_dataframe()['y'].values.tolist()
        x_std = [1]*k

        return x, x_std
        
    def darts_Theta_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, deseasonalize=False):
        """
            Calls Darts' Theta forecasting model, but ignores t if supplied.
        """
        if a:
            assert len(a) == len(y) + k
        if np.isscalar(y[0]):
            y0s = [ yt for yt in y]
        else:
            y0s = [ yt[0] for yt in y ]

        y0s.extend([0]*k)
        idx = pd.date_range(end='01/01/2000', periods=len(y0s), freq="H")
        df = pd.DataFrame(y0s, index=idx).reset_index().rename(columns={'index':'ts',0:'y'})
        df.head()

        series = TimeSeries.from_dataframe(df, 'ts', 'y')
        train, val = series[:-k], series[-k:]

        model = Theta(season_mode = SeasonalityMode.ADDITIVE)
        model.fit(train)
        prediction = model.predict(len(val))

        x = prediction.pd_dataframe()['y'].values.tolist()
        x_std = [1]*k

        return x, x_std
        
    def darts_FourTheta_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, deseasonalize=False):
        """
            Calls Darts' FourTheta forecasting model, but ignores t if supplied.
        """
        if a:
            assert len(a) == len(y) + k
        if np.isscalar(y[0]):
            y0s = [ yt for yt in y]
        else:
            y0s = [ yt[0] for yt in y ]

        y0s.extend([0]*k)
        idx = pd.date_range(end='01/01/2000', periods=len(y0s), freq="H")
        df = pd.DataFrame(y0s, index=idx).reset_index().rename(columns={'index':'ts',0:'y'})
        df.head()

        series = TimeSeries.from_dataframe(df, 'ts', 'y')
        train, val = series[:-k], series[-k:]

        model = FourTheta(season_mode = SeasonalityMode.ADDITIVE)
        model.fit(train)
        prediction = model.predict(len(val))

        x = prediction.pd_dataframe()['y'].values.tolist()
        x_std = [1]*k

        return x, x_std

    def darts_NBEATS_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, deseasonalize=False):
        """
            Calls Darts' NBEATS forecasting model, but ignores t if supplied.
        """
        if a:
            assert len(a) == len(y) + k
        if np.isscalar(y[0]):
            y0s = [ yt for yt in y]
        else:
            y0s = [ yt[0] for yt in y ]

        y0s.extend([0]*k)
        idx = pd.date_range(end='01/01/2000', periods=len(y0s), freq="H")
        df = pd.DataFrame(y0s, index=idx).reset_index().rename(columns={'index':'ts',0:'y'})
        df.head()

        series = TimeSeries.from_dataframe(df, 'ts', 'y')
        train, val = series[:-k], series[-k:]

        model = NBEATSModel(input_chunk_length = 2 * k, output_chunk_length = k)
        model.fit(train)
        prediction = model.predict(len(val))

        x = prediction.pd_dataframe()['y'].values.tolist()
        x_std = [1]*k

        return x, x_std
        

if __name__=='__main__':
    assert using_darts, 'pip install \'u8darts[all]\''
    import numpy as np
    import time
    st = time.time()
    y = list(np.cumsum(np.random.randn(400)))
    x, x_std = darts_FFT_iskater(y=y, k=5)
    print(x)
    print(time.time()-st)