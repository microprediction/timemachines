from timemachines.skaters.ik.ikinclusion import using_ik

if using_ik:
    import numpy as np
    import joblib
    from typing import List


    def ik_nnma3_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, deseasonalize=False):
        """
            Calls nueral network implementation of a moving average 
            forecasting that should be a function of the last r data points.
        """
        if np.isscalar(y[0]):
            y0s = [ yt for yt in y]
        else:
            y0s = [ yt[0] for yt in y ]

        if len(y0s) < 100:
            return [y0s[-1]] * k, [1] * k

        nn = joblib.load('nn_ma3.pkl')
        pred = nn.predict(np.array(y0s[-10:]).reshape(1,-1))

        x = [pred] * k
        x_std = [1] * k

        return x, x_std

    def ik_nnma10_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, deseasonalize=False):
        """
            Calls nueral network implementation of a moving average 
            forecasting that should be a function of the last r data points.
        """
        if np.isscalar(y[0]):
            y0s = [ yt for yt in y]
        else:
            y0s = [ yt[0] for yt in y ]

        if len(y0s) < 100:
            return [y0s[-1]] * k, [1] * k

        nn = joblib.load('nn_ma10.pkl')
        pred = nn.predict(np.array(y0s[-10:]).reshape(1,-1))

        x = [pred] * k
        x_std = [1] * k

        return x, x_std

    def ik_nnma100_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, deseasonalize=False):
        """
            Calls nueral network implementation of a moving average 
            forecasting that should be a function of the last r data points.
        """
        if np.isscalar(y[0]):
            y0s = [ yt for yt in y]
        else:
            y0s = [ yt[0] for yt in y ]

        if len(y0s) < 100:
            return [y0s[-1]] * k, [1] * k

        nn = joblib.load('nn_ma100.pkl')
        pred = nn.predict(np.array(y0s[-100:]).reshape(1,-1))

        x = [pred] * k
        x_std = [1] * k

        return x, x_std


if __name__=='__main__':
    assert using_ik, 'pip install -U scikit-learn'
    import numpy as np
    import time
    st = time.time()
    y = list(np.cumsum(np.random.randn(400)))
    x, x_std = ik_nnma3_iskater(y=y, k=5)
    print(x)
    print(time.time()-st)