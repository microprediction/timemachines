from timemachines.skaters.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE, wrap
from typing import Any
from timemachines.skaters.components.parade import parade


def residual_chaser_factory(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None,
                            f1=None, f2=None, chase=1.0, threshold=1.0, r1=None, r2=None)->([float] , Any , Any):
    """ Last value cache, with empirical std and self-correction

          f1  - A skater making the primary prediction
          f2  - A skater designed to predict residuals
          chase     - Fraction of f2's residual prediction to use
          threshold - Number of standard deviations the residual prediction must exceed before we chase it.
          r1  - hyper-params for f1, if any
          r2  - hyper-params for f2, if any

    """
    y0 = wrap(y)[0]
    if not s.get('p1'):
        s = {'p1': {},
             'x': y0,
             's1':{},
             's2':{},
             'n_obs':0}

    if y0 is None:
        return None, None, s
    else:
        # Use the first skater to predict
        if r1 is None:
            x1, x1_std, s['s1'] = f1(y=y,s=s['s1'],k=k, a=a,t=t,e=e)
        else:
            x1, x1_std, s['s1'] = f1(y=y, s=s['s1'], k=k, a=a, t=t, e=e, r=r1)
        x1_error_mean, x1_error_std, s['p1'] = parade(p=s['p1'], x=x1, y=y0)  # Update prediction queue
        s['n_obs']+=1

        # Use the second skater to predict mean residual k-steps ahead
        xke = x1_error_mean[-1]
        if r2 is None:
            xke_hat_mean, xke_hat_std, s['s2'] = f2(y=[xke],s=s['s2'],k=k,a=a,t=t,e=e)
        else:
            xke_hat_mean, xke_hat_std, s['s2'] = f2(y=[xke], s=s['s2'], k=k, a=a, t=t, e=e, r=r2)

        # If the bias prediction is confident, adjust x1 chasing it towards the bias corrected value
        if s['n_obs']>10:
            for j in range(len(x1)):
                if abs(xke_hat_mean[j])>threshold*xke_hat_std[j]:
                    x1[j] = x1[j]+chase*xke_hat_mean[j]

        return x1, x1_std, s









