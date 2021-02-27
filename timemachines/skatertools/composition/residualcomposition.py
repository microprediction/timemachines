from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, E_TYPE, T_TYPE, wrap
from typing import Any
from timemachines.skatertools.components.residuals import residual



def residual_chaser_factory(y :Y_TYPE, s:dict, k:int, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None,
                            f1=None, f2=None, r1=None, r2=None)->([float] , Any , Any):
    """ Second model predicts k=1, k=k residuals of the first, and interpolates

          f1  - A skater making the primary prediction
          f2  - A skater designed to predict residuals ... both 1 step ahead and k-steps ahead
          r1  - hyper-params for f1, if any
          r2  - hyper-params for f2, if any

       It *may* make sense to choose an f2 that shrinks towards zero.
    """
    if k == 1:
        J = [1]
    else:
        J = [1,k]  # Determines horizons over which residual model is used.
                   # We'd rather not call the residual model k-times

    y0 = wrap(y)[0]
    if not s.get('s1'):
        s = {'sres': {},                      # Residual state ... used to determine the residual
             'x': y0,
             's1':{},                         # First model state
             's2':dict([(j,{}) for j in J]),  # Residual model states
             'n_obs':0}

    if y0 is None:
        return None, None, s
    else:
        # Use the first skater to predict
        if r1 is None:
            x1, x1_std, s['s1'] = f1(y=y,s=s['s1'],k=k, a=a,t=t,e=e)
        else:
            x1, x1_std, s['s1'] = f1(y=y, s=s['s1'], k=k, a=a, t=t, e=e, r=r1)
        resid1, s['sres'] = residual(s['sres'],y=y0,x=x1)

        s['n_obs']+=1

        # Use the second skater to predict j-step ahead residuals
        # There are two copies of the residual model employed.
        res_j_hat = [None for j in J]
        res_j_std = [None for j in J]
        for jpos,j in enumerate(J):
            j_ahead_residual = resid1[j-1]
            if r2 is None:
                _x, _std, s['s2'][j] = f2(y=j_ahead_residual, s=s['s2'][j], k=j, a=a, t=t, e=e)
            else:
                _x, _std, s['s2'][j] = f2(y=j_ahead_residual, s=s['s2'][j], k=j, a=a, t=t, e=e,r=r2)
            res_j_hat[jpos] = _x[jpos]
            res_j_std[jpos] = _std[jpos]

        # Interpolate
        if k==1:
            res_interp = res_j_hat
            res_interp_std = res_j_std
        else:
            import numpy as np
            ks = list(range(1,k+1))
            res_interp = np.interp( x=ks, xp=J, fp=res_j_hat )
            res_interp_std = np.interp(x=ks, xp=J, fp=res_j_std)

        # Residual   res =  y - x1,   so  x1+res ~ y  .... one hopes
        x_hat = [ resj+x1j for resj, x1j in zip( res_interp, x1) ]
        return x_hat, res_interp_std, s









