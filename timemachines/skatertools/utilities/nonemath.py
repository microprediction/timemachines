from typing import Union
from numpy import asarray
from itertools import zip_longest


def nonelen(l):
    return 0 if l is None else len(l)


def noneratio(x,y):
    return None if x is None or y is None else x/y


def nonecast(x,fill_value=0.0):
    if x is not None:
        return [xj if xj is not None else fill_value for xj in x]


def notallnone(x):
    return not all([ xj is None for xj in x])


def nearlysame(x,y,tol=1e-6):
    return all( ( abs(xj-yj)<tol for xj,yj in zip_longest(x,y,fillvalue=.777)))


def nonennearlysame(x,y,tol=1e-6):
    x_cast = nonecast(x, fill_value=333.0)
    y_cast = nonecast(x, fill_value=333.0)
    return nearlysame(x_cast,y_cast)


def nonecenter(m,x):
    """
    :param m:  List of masses
    :param x:  List of points, can be None or vector
    :returns: point that is center of mass
    """
    assert notallnone(m)
    assert notallnone(x)
    m_when_x_not_none = [ m for m,x in zip(m,x) if x is not None]
    x_when_x_not_none = [ x for m,x in zip(m,x) if x is not None]
    m_cast = nonecast(m_when_x_not_none,fill_value=0.0)
    x_cast = [ nonecast(pj,fill_value=0.0) for pj in x_when_x_not_none]
    if sum(m_cast)<1e-9:
        m_cast = [1./len(m_cast) for _ in m_cast ]
    return center(m_cast, x_cast)


def center(m:[float], x:[[float]])->[Union[None, float]]:
    """ Center of mass
          m   masses
          x   set of points
    """
    if len(m)!=len(x):
        import warnings
        warnings.warn('length mismatch')
    P = asarray([ asarray(xj) for xj in x])
    M = asarray(m)
    PM = (P.T * M).T
    try:
        PM_sum = PM.sum(axis=0)
    except:
        k = 1
        pass
    c = PM_sum / M.sum()
    return c.tolist()

