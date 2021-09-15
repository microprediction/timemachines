from timemachines.skaters.tcn.tcninclusion import using_tcn

if using_tcn:
    import numpy as np
    import pandas as pd
    import datetime
    from typing import List
    from timemachines.skatertools.utilities.suppression import no_stdout_stderr

    def tcn_univariate_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, tnc_onnx_model=None):
        """

        """
        if a:
            assert len(a) == len(y) + k
        if np.isscalar(y[0]):
            y0s = [ yt for yt in y]
        else:
            y0s = [ yt[0] for yt in y ]

        

        x_std = [1.0 for _ in x]
        return x, x_std


if __name__=='__main__':
    assert using_tcn, 'pip install keras'
    import numpy as np
    import time
    st = time.time()
    y = list(np.cumsum(np.random.randn(400)))
    x, x_std = tcn_univariate_iskater(y=y, k=5)
    print(x)
    print(time.time()-st)
