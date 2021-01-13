from typing import List, Iterator


def reflect(ys:[float], n:int)->[float]:
    """ Lengthen time series by reflection back in time
    :param ys: time series shorter than n
    :param n:  desired length of time series
    :return: time series length n
    """
    assert len(ys)>=3
    xs = list(reversed(ys))[:-1] + list(ys)
    while len(xs)<n:
        xs = xs[:-1] + xs
    return xs[:n]


def dilate(u,scale):
    if isinstance(u,list):
        return [u_*scale for u_ in u]
    else:
        return u*scale


if __name__=='__main__':
    import random
    ys = [1,2,3,4,5,6,7,8,9,10]
    print(ys)
    print( chunk_to_end(ys,3) )