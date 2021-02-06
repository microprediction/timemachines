import numpy as np


def brownian_with_noise(n:int)->[float]:
    ys_ = np.cumsum(np.random.randn(n))
    ys = [y + np.random.randn() for y in ys_]
    return list(ys)


def brownian_with_exogenous(n:int)->[(float,)]:
    dys1__ = np.random.randn(n+2)
    dys1_ = np.array([ 0.1*a+0.3*b+0.2*c for a,b,c in zip( dys1__[2:], dys1__[1:],dys1__) ])
    dys2_ = list(0.5*np.array(dys1_[2:]))+list(np.random.randn(2))
    ys1 = np.cumsum(dys1_) + 0.025*np.random.randn(n)
    ys2 = np.cumsum(dys2_+0.1*dys1_)+0.15*ys1 + 0.015*np.random.randn(n)
    return list(zip(ys1,ys2))


