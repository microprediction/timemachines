

def residual(s:dict, y, x):
    """ Return residuals
      :param s:   state - supply empty dict on first call
      :param y:   incoming observation
      :param x:   term structure of predictions out k steps ahead, made after y received
      :returns k-vector of residuals
    """
    k = len(x)
    if not s:
        s = {'predictions': [[] for _ in range(k)]}  # Holds the cavalcade
    else:
        assert len(x) == len(s['predictions'])  # 'k' is immutable

    assessable = s['predictions'].pop(0)
    z = [0 for _ in range(k)]
    if assessable:
        for j, xi in assessable:
            z[j] = y - xi

    s['predictions'].append(list())
    for j, xj in enumerate(x):
        s['predictions'][j].append((j, xj))

    return z, s