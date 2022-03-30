from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, E_TYPE, T_TYPE, wrap
from typing import Any, List
from timemachines.skatertools.ensembling.precisionweightedskater import precision_weighted_skater
from timemachines.skatertools.components.parade import parade
from timemachines.skatertools.utilities.nonemath import noneupdatelist


# Tools for


# Some r values to use when ensembling
R_PRECISION = 0.5    # (i.e. precision weighting)
R_BALANCED = 0.25
R_AGGRESSIVE = 0.90


def ensemble_factory(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None,
                     fs: List = None, rs: List = None, g=None, r=None, include_std=True, trust=True,
                     empirical_std=False) -> ([float], Any, Any):
    """ Meta-model skater

          fs  - list of skaters
          rs  - list of hyper-params, if any
          g   - exogenous skater (the meta model)
          r   - hyper-param for g, if any
          include_std   - If True, will interleave x_std in the exogenous variables sent to g
          trust         - If True will use the models' standard deviation, otherwise empirical
          empirical_std - If True will use empirical std, otherwise leans on models' estimates

        It is possible to change the list of skaters supplied every now and then but be aware this will
        throw off the tracking of empirical errors in the parade a little.

    """
    if not s.get('s_fs'):
        s = {'s_fs': dict([(f.__name__,{}) for f in fs]),  # State for skaters
             's_g': {},                 # State for meta-model
             'n_obs':0,
             's_fs_p':dict([(f.__name__,{}) for f in fs]), # Holds parades for skaters
             's_p':{},                  # Self parade
             's_w':''                   # Holds current winning model
             }

    if y is None:
        return None, None, s
    else:
        # New f introduced?
        for f in fs:
            if not f.__name__ in s['s_fs']:
                s['s_fs'][f.__name__]={}
                s['s_fs_p'][f.__name__]={}

        # Apply models, keeping only the point estimate
        # This constructs a flat list of child predictions and their standard deviations,
        # but for now only uses the k-step ahead predictions to judge, not k-1, k-2,...
        xjs = list()
        rs = rs or [None for _ in fs]
        y0 = wrap(y)[0]
        s['s_w'] = fs[0].__name__
        lowest_std = 1e14
        for j, (f, r) in enumerate(zip(fs, rs)):
            if r is not None:
                xj, xj_std, s['s_fs'][f.__name__] = f(y=y, s=s['s_fs'][f.__name__], k=k, a=a, t=t, e=e, r=r)
            else:
                xj, xj_std, s['s_fs'][f.__name__] = f(y=y, s=s['s_fs'][f.__name__], k=k, a=a, t=t, e=e)
            xjs.append(xj[-1])  # <--- Only use the k-step forward prediction and toss out the rest (a bit lazy)

            if not trust:
                _, emp_std, s['s_fs_p'][f.__name__] = parade(p=s['s_fs_p'][f.__name__], x=xj, y=y0 )
                xj_std = noneupdatelist(xj_std, emp_std)

            if xj_std[-1] < lowest_std:
                s['s_w'] = f.__name__
                lowest_std = xj_std[-1]
            if include_std:
                xjs.append(xj_std[-1])

        # Next we'll pass the child predictions to a combining meta-model
        s['n_obs']+=1
        if s['n_obs']<10:
            return [wrap(y)[0]]*k, [wrap(y)[0]]*k, s
        else:
            y_extend = [wrap(y)[0]] + xjs
            if r is None:
                x, x_std, s['s_g'] = g(y=y_extend, s=s['s_g'], k=k, a=a, t=t, e=e)
            else:
                x, x_std, s['s_g'] = g(y=y_extend, s=s['s_g'], k=k, a=a, t=t, e=e, r=r)
            if empirical_std:
                _, self_emp_std, s['s_p'] = parade(p=s['s_p'], x=x, y=y0)
                x_std = noneupdatelist(x_std, self_emp_std)
            return x, x_std, s


def precision_weighted_ensemble_factory(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None,
                     fs: List = None, rs: List = None, r=None, trust=False, empirical_std=True) -> ([float], Any, Any):
    """
             r -   determines the exponent
    """
    g = precision_weighted_skater
    return ensemble_factory(y=y,s=s,k=k,a=a,t=t,e=e,fs=fs,rs=rs,g=g,r=r, trust=trust, empirical_std=empirical_std)


def trusting_precision_weighted_ensemble_factory(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None,
                     fs: List = None, rs: List = None, r=None, trust=True, empirical_std=True) -> ([float], Any, Any):
    """
             r -   determines the exponent
    """
    g = precision_weighted_skater
    return ensemble_factory(y=y,s=s,k=k,a=a,t=t,e=e,fs=fs,rs=rs,g=g,r=r, trust=trust, empirical_std=empirical_std)
