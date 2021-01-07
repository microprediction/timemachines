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


def chunk_to_end(l:List, n:int)-> List[List]:
    """
        :param n: Size of batches
    """
    rl = list(reversed(l))
    chunks = [ list(reversed(rl[x:x + n])) for x in range(0, len(rl), n) ]
    return list(reversed(chunks[:-1]))



if __name__=='__main__':
    import random
    ys = [1,2,3,4,5,6,7,8,9,10]
    print(ys)
    print( chunk_to_end(ys,3) )