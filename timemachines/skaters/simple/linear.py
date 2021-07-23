from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, E_TYPE, T_TYPE, wrap
from typing import Any
import numpy as np
import math
from timemachines.skatertools.components import observance
from timemachines.skatertools.components.parade import parade


def regress_change_on_first_known(y:Y_TYPE, s:dict, k, a:A_TYPE=None, t:T_TYPE =None, e:E_TYPE =None )->([float] , Any , Any):
    """ Very basic modification of the last value cache.
        This looks at the contemporaneous influence of a single known in advance variable.
        Assumes independent increments when estimating the standard deviation.
        This is also intended to illustrate combination of skaters
    """
    y0 = wrap(y)[0]  # Ignore contemporaneous, exogenous variables
    if not s.get('prev_y0'):
        s = {'prev_y0':y0,
             'd':{}   # state for difference predicting skater
             }
        return y, 1.0, s
    else:
        dy0 = y0 - s['prev_y0']
        dy_hat, dy_hat_std = regress_level_on_first_known(y=[dy0], s=s['d'], k=k, a=a, t=t, e=e)
        x = [y0 + sum_dy for sum_dy in np.cumsum(dy_hat)]
        x_std = [ math.sqrt(v) for v in np.cumsum([ s**s for s in dy_hat_std])]
        return x, x_std, s


def regress_level_on_first_known(y:Y_TYPE, s:dict, k, a:A_TYPE=None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
    """ Very basic online regression skater, mostly for offlinetesting
           - Only one known in advance variable is utilized
           - Last value is ignored, unless a is None in which case we return 0.0
           - Empirical std is returned
    """
    y0 = wrap(y)[0]  # Ignore contemporaneous, exogenous variables
    if a:
        a0 = wrap(a)[0]  # Ignore all but the first known-in-advance variable

    if not s.get('k'):
        # First invocation
        s = {'p': {}} # Prediction parade
        s['r'] = {}   # Regression state, not to be confused with hyper-param r
        s['k'] = k
        s['o'] = {}   # The "observance" will quarantine 'a' until it can be matched
    else:
        assert s['k']==k  # Immutability

    if a is None:
        return [0]*k, [1.0]*k, s
    else:
        a_t, s['o'] = observance( y=[y0],o=s['o'], k=k, a= [a0])  # Update the observance
        if a_t is not None: # This is the contemporaneous 'a', which was supplied k calls ago.
            if not s['r']:
                # When first calling the online regression algorithm we avoid the degenerate case
                # by sending it two observations.
                y_noise = 0.1*(1e-6+abs(y0))*np.random.randn()
                x_noise = 0.1*(1e-6+abs(a0))*np.random.randn()
                x = [ a_t[0]-x_noise, a_t[0]+x_noise  ]
                y = [ y0-y_noise,  y0+y_noise  ]
                s['r'] = regress_one_helper(x=x, y=y, r=s['r'])
            else:
                s['r'] = regress_one_helper(x=a_t, y=[y0], r=s['r'])

            # Predict using contemporaneous alpha's
            x = [ s['r']['alpha'] + s['r']['beta']*ak[0] for ak in s['o']['a'] ]

            # Push prediction into the parade and get the current bias/stderr
            bias, x_std, s['p'] = parade(p=s['p'], x=x, y=y0)
            return x, x_std, s    # TODO: Use the std implied by regression instead
        else:
            x = [y0]*k
            bias, x_std, s['p'] = parade(p=s['p'], x=x, y=y0)
            return x , x_std, s


LINEAR_SKATERS = [regress_change_on_first_known]


def regress_one_helper(x: [float], y: [float], r:dict)->dict:
    """
        Helper providing online regression for scalar y and a single regressor x.
        Updates a batch of observations.
        From https://stackoverflow.com/questions/52070293/efficient-online-linear-regression-algorithm-in-python

        r - Regression state

                x_avg: average of previous x, if no previous sample, set to 0
                y_avg: average of previous y, if no previous sample, set to 0
                Sxy: covariance of previous x and y, if no previous sample, set to 0
                Sx: variance of previous x, if no previous sample, set to 0
                n: number of previous samples
                new_x: new incoming 1-D numpy array x
                new_y: new incoming 1-D numpy array x
    """
    if not r.get('n_obs'):
        r = {'n_obs': 0,
             'x_avg': 0,
             'y_avg': 0,
             'Sxy': 0,
             'Sx': 0}

    n, x_avg, y_avg, Sxy, Sx = r['n_obs'], r['x_avg'], r['y_avg'], r['Sxy'], r['Sx']

    new_n = n + len(x)
    new_x_avg = (x_avg * n + np.sum(x)) / new_n
    new_y_avg = (y_avg * n + np.sum(y)) / new_n
    if n > 0:
        x_star = (x_avg * np.sqrt(n) + new_x_avg * np.sqrt(new_n)) / (np.sqrt(n) + np.sqrt(new_n))
        y_star = (y_avg * np.sqrt(n) + new_y_avg * np.sqrt(new_n)) / (np.sqrt(n) + np.sqrt(new_n))
    elif n == 0:
        x_star = new_x_avg
        y_star = new_y_avg
    else:
        raise ValueError
    new_Sx = Sx + np.sum((x - x_star) ** 2)
    new_Sxy = Sxy + np.sum((x - x_star).reshape(-1) * (y - y_star).reshape(-1))
    beta = new_Sxy / new_Sx
    alpha = new_y_avg - beta * new_x_avg

    s_posterior = {'Sxy':new_Sxy,'Sx':new_Sx,'n_obs':new_n,'x_avg':new_x_avg,'y_avg':new_y_avg,
                   'alpha':alpha,'beta':beta}

    return s_posterior



if __name__=='__main__':
    # A little test...
    true_beta = 2.0
    a = [ i % 5 for i in range(500) ]
    y = [ true_beta*aj+np.random.randn() for aj in a ]
    s, k = {}, 3
    for aj, yj in zip(a[k:],y):
        x, x_std, s = regress_level_on_first_known(y=yj, s=s, k=k, a=aj)
    assert abs(s['r']['beta']-true_beta)<0.1, 'You broke it somehow! '




