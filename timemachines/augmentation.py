
# Trivial data augmentation
import numpy as np


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


if __name__=='__main__':
    ys = np.cumsum(np.random.randn(15))
    xs = reflect(ys,170)
    import matplotlib.pyplot as plt
    plt.plot(xs)
    plt.show()