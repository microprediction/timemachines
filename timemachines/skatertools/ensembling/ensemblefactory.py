from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, E_TYPE, T_TYPE, wrap
from typing import Any, List
from timemachines.skatertools.ensembling.precisionweightedskater import precision_weighted_skater


def ensemble_factory(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None,
                     fs: List = None, rs: List = None, g=None, r=None, include_std=True) -> ([float], Any, Any):
    """ Ensembles *only* the k-step ahead

          fs  - list of skaters
          rs  - list of hyper-params, if any
          g   - exogenous skater
          r   - hyper-param for g, if any
          include_std - bool. If True, will add x_std to the exogenous variables sent to g

    """
    if not s.get('s_fs'):
        s = {'s_fs': [{} for _ in fs],
             's_g': {},
             'n_obs':0}

    if y is None:
        return None, None, s
    else:
        # Apply models, keeping only the point estimate
        # This constructs a flattenned list of child predictions and their standard deviations,
        # but for now only uses the k-step ahead predictions, not k-1, k-2,...
        xjs = list()
        rs = rs or [None for _ in fs]
        for j, (f, r) in enumerate(zip(fs, rs)):
            if r is not None:
                xj, xj_std, s['s_fs'][j] = f(y=y, s=s['s_fs'][j], k=k, a=a, t=t, e=e, r=r)
            else:
                xj, xj_std, s['s_fs'][j] = f(y=y, s=s['s_fs'][j], k=k, a=a, t=t, e=e)
            xjs.append(xj[-1])  # <--- Only use the k-step forward prediction and toss out the rest (a bit lazy)
            if include_std:
                xjs.append(xj_std[-1])

        # Next we'll pass the child predictions to a stateless combining model
        s['n_obs']+=1
        if s['n_obs']<10:
            return [wrap(y)[0]]*k, [wrap(y)[0]]*k, s
        else:
            y_extend = [wrap(y)[0]] + xjs
            if r is None:
                x, x_std, s['s_g'] = g(y=y_extend, s=s['s_g'], k=k, a=a, t=t, e=e)
            else:
                x, x_std, s['s_g'] = g(y=y_extend, s=s['s_g'], k=k, a=a, t=t, e=e, r=r)
            return x, x_std, s


def precision_weighted_ensemble_factory(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None,
                     fs: List = None, rs: List = None, r=None) -> ([float], Any, Any):
    """
             r -   determines the exponent
                   r=0.5 corresponds to simple weighting, and is the default
    """
    g = precision_weighted_skater
    return ensemble_factory(y=y,s=s,k=k,a=a,t=t,e=e,fs=fs,rs=rs,g=g,r=r)

