import numpy as np


def frac_weights(d: float, m: int = 20) -> list:
    """
    :param d:  Fractional order
    :param m:  Weights are length m+1
    :return:
    """
    w = [1.0]
    for k in range(1, m + 1):
        w.append(-w[-1] * ((d - k + 1.0) / k))
    return w


def frac_matrix(d: float, m:int = 20, n:int=50, l:int=15):
    """
    :param d:
    :param m: Length of weights
    :param n: Total length of the time-series
    :param l: The number of differences that will be taken (n-l will be left alone)
    :return:
    """
    assert m + 1 + l <= n

    w = frac_weights(d=d,m=m)
    A = np.eye(n)
    for j in range(0,l):
        A[j,j:j+m+1] = w
    return A


if __name__=='__main__':
    d = 0.5
    m = 10
    l = 15
    n = 50
    A = frac_matrix(d=d,m=m, n=n, l=l)
    print(A)
    B = np.linalg.inv(A)
    C = np.matmul(A,B)
    print(C)
