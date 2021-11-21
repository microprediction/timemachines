from timemachines.skaters.ik.ikinclusion import using_ik

if using_ik:
    import numpy as np
    import pandas as pd
    import joblib
    from typing import List


    def ik_nnma_iskater(y: [[float]], k: int, t: float, r: float):
        """
            Calls nueral network implementation of a moving average 
            forecasting that should be a function of the last r data points.
        """
        if np.isscalar(y[0]):
            y0s = [ yt for yt in y]
        else:
            y0s = [ yt[0] for yt in y ]

        if len(y0s) < t or len(y0s) < r:
            return [y0s[-1]] * k, [1] * k

        if r < 100:
            num_cols = 10
        else:
            num_cols = 100

        col_names = []
        for i in range(0,num_cols):
            col_names.append('t'+str(i))
        data = [y0s[-num_cols:]]
        df_ts = pd.DataFrame(data, columns = col_names)

        nn = joblib.load('nn_ma'+str(int(r))+'.pkl')
        pred = nn.predict(df_ts)

        x = [pred] * k
        x_std = [1] * k

        return x, x_std


if __name__=='__main__':
    assert using_ik, 'pip install -U scikit-learn'
    import numpy as np
    import time
    st = time.time()
    y = list(np.cumsum(np.random.randn(400)))
    x, x_std = ik_nnma_iskater(y=y, k=5, t=10, r=3)
    print(x)
    print(time.time()-st)