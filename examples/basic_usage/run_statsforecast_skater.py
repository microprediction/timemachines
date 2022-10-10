try:
    from timemachines.skaters.sk.sfautoarima import sf_autoarima as f
except ImportError:
    print('pip install timemachines')
    print('pip install sktime')
    print('pip install statsmodels')

import numpy as np

# Example of running Nixtla statsforecast AutoARIMA sequentially over 100 data points

if __name__=='__main__':
    y = np.cumsum(np.random.randn(100))
    s = {}
    x = list()
    for yi in y:
        xi, x_std, s = f(y=yi, s=s, k=3, e=1000)
        x.append(xi)
