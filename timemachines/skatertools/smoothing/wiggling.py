import itertools
from momentum import var_init, var_update
import numpy as np
from timemachines.skatertools.components.parade import parade
from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, E_TYPE, T_TYPE, wrap
from timemachines.skatertools.utilities.nonemath import nonemax

# The "Wiggler"


def ternary_product(m):
    the_set = {-1,0,1}
    return [ list(s) for s in itertools.product(the_set, repeat=m) ]


def wiggler(f, y, s, k=1, m:int=5, d=0.1, track=False, **kwargs):
    """ Modifies a skater f to create one that depends more smoothly on the history

        This maintains m^3 copies of f that are fed slightly different observations
        It returns a simple average of the predictions.

    :param m:  Memory. Warning this will create m^3.
    :param d:  The difference multiplier (how much to bump points as a multiple of emp standard deviation)
    :param track: boolean If True will keep accuracy statistics for individual bumps - not implemented yet
    :param **kwargs  Optional a, t, e, r passed to skater f
    :return:
    """

    if not s.get('s_f'):
        assert int(m) == m
        assert m <= 10, 'Not recommended for memory this long'
        assert m >= 1
        patterns = ternary_product(m=m)
        n_patterns = len(patterns)

        s = {
             'patterns':patterns,
             'm':m,
             'n_patterns':n_patterns,
             's_f':dict(),
             's_fs': [ dict() for _ in range(n_patterns) ],   # State for skaters
             'n_': 0,
             's_fs_p':[ dict() for _ in range(n_patterns) ],  # Holds parades for skaters
             's_f_p': {},  # Self parade
             's_dy_var':var_init(),
             'y_prev':None
             }

    if y is None:
        return None, None, s
    else:
        # Apply the base skater and track its error
        y0 = wrap(y)[0]
        s_f = s['s_f']
        x, x_std, s_f = f(y=y, s=s_f, k=k, **kwargs )
        s['s_f'] = s_f
        _, emp_std, s['s_f_p'] = parade(p=s['s_f_p'], x=x, y=y0)

        # Update running std of differences
        if s['y_prev'] is not None:
            dy = y-s['y_prev']
            s['y_prev'] = y
            s['s_dy_var'] = var_update(s['s_dy_var'], dy)
        dy_std =  s['s_dy_var'].get('std')

        bump_amount = nonemax([emp_std[0],dy_std,0])

        # Loop over patterns to get predictions
        # Update the patterns as well as the state
        xs = [ [] for _ in range(s['n_patterns']) ]
        x_stds = [ [] for _ in range(s['n_patterns']) ]
        for i, (pattern, state_i) in enumerate(zip(s['patterns'],s['s_fs'])):
            sgn = pattern[0]
            bump = d*sgn*bump_amount
            new_pattern = pattern[1:] + [pattern[0]] # rotate
            y_bumped = y + bump
            x, x_std, state_i = f(y=y_bumped, s=state_i, k=k, **kwargs)
            s['s_fs'][i] = state_i
            xs[i] = x
            x_stds[i] = x_std
            s['patterns'][i] = new_pattern

        # Average the prediction vectors
        try:
            avg_xs = list(np.average( np.array(xs), axis=0 ))
            avg_xstd = list(np.average( np.array(x_stds), axis=0 ) )
        except TypeError:
            print(xs)
        return avg_xs, avg_xstd, s



if __name__=='__main__':
    S = ternary_product(m=3)
    from pprint import pprint
    pprint(S)