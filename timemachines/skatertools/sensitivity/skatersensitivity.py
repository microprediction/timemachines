from copy import deepcopy
from timemachines.skatertools.data.ornstein import simulate_arima_like_path


def skater_bump(ys, f, num_points=51, ndx=-1, k=1):
    """ Sensitivity to last value  """
    # Pass e<0 for most of the data
    assert ndx<0
    import numpy as np
    y_bumped = [ ys[ndx] + 0.01 * i for i in np.linspace(-100,100,num_points)]

    # Roll forward to -ndx
    s = {}
    for y in ys[:ndx]:
        x, x_std, s = f(y=y, k=k, s=s, e=-1)

    # Process bumped observations
    x_final_values = list()
    for y_bump in y_bumped:
        s_ = deepcopy(s)
        # Process ndx, which might be the last
        e = 1000 if ndx==-1 else -1
        x, x_std, s_ = f(y=y_bump, k=k, s=s_, e=e)
        # Might need to process from ndx+1 to -2
        for y in ys[ndx:-2]:
            x, x_std, s_ = f(y=y, k=k, s=s_, e=-1)
        # Process the last with e=1000
        if ndx<-1:
            x, x_std, s_ = f(y=ys[-1], k=k, s=s_, e=1000)
        x_final_values.append(x[k-1])

    return y_bumped, x_final_values

