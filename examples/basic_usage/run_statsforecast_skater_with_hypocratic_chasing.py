try:
    from timemachines.skaters.sk.sfautoarimahypocratic import sf_autoarima_hypocratic as f
except ImportError:
    print('pip install timemachines')
    print('pip install sktime')
    print('pip install statsforecast')

import numpy as np

# Example of running Nixtla statsforecast AutoARIMA sequentially over 100 data points, and chasing residuals. 

if __name__=='__main__':
    y = np.cumsum(np.random.randn(100))
    s = {}
    x = list()
    for yi in y:
        xi, x_std, s = f(y=yi, s=s, k=3, e=1000)
        x.append(xi)
