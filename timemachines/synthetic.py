import numpy as np


def brownian_with_noise(n:int)->[float]:
    ys_ = np.cumsum(np.random.randn(n))
    ys = [y + np.random.randn() for y in ys_]
    return list(ys)


def brownian_with_exogenous(n:int)->[(float,)]:
    dys1_ = np.random.randn(n)
    dys2_ = list(0.5*dys1_[2:])+list(np.random.randn(2))
    ys1 = np.cumsum(dys1_) + 0.15*np.random.randn(n)
    ys2 = np.cumsum(dys2_+0.1*dys1_)+0.15*ys1 + 0.15*np.random.randn(n)
    return list(zip(ys1,ys2))


