from timemachines.skaters.simple.thinking import thinking_slow_and_fast
import numpy as np


if __name__=='__main__':
    y = np.cumsum(np.random.randn(1000))
    s = {}
    x = list()
    for yi in y:
        xi, x_std, s = thinking_slow_and_fast(y=yi, s=s, k=3)
        x.append(xi)