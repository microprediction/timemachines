from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE, wrap
from typing import Any
from timemachines.skatertools.components.parade import parade
from timemachines.skaters.nproph.nprophiskaterfactory import nprophet_iskater_factory
from timemachines.skatertools.utilities.nonemath import nonecenter
from timemachines.skaters.nproph.nprophparams import NPROPHET_META, nprophet_params
from timemachines.skatertools.utilities.nonemath import nonecast
import sys
import logging
import numpy as np

logging.disable(sys.maxsize)
logging.getLogger('fbnprophet').setLevel(logging.DEBUG)


def fbnprophet_skater_factory(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                             t: T_TYPE = None, e: E_TYPE = None,
                             emp_mass: float = 0.0, emp_std_mass: float = 0.0,
                             freq=None, recursive: bool = False,
                             model_params: dict = None,
                             n_max: int = None) -> ([float], Any, Any):
    """ Prophet skater with running prediction error moments
        Hyper-parameters are explicit here, whereas they are determined from r in actual skaters.
        Params of note:

             a: value of known-in-advance vars k step in advance (not contemporaneous with y)

    """

    assert 0 <= emp_mass <= 1
    assert 0 <= emp_std_mass <= 1

    if freq is None:
        freq = NPROPHET_META['freq']
        
    if n_max is None:
        n_max = NPROPHET_META['n_max']

    y = wrap(y)
    a = wrap(a)

    if not s.get('y'):
        s = {'p': {},     # parade
             'y': list(), # historical y
             'a': list(), # list of a known k steps in advance
             't': list(),
             'k': k}
    else:
        # Assert immutability of k, dimensions of y,a
        if s['y']:
            assert len(y) == len(s['y'][0])
            assert k == s['k']
        if s['a']:
            assert len(a) == len(s['a'][0])

    if y is None:
        return None, s, None
    else:
        s['y'].append(y)
        if a is not None:
            s['a'].append(a)
        if t is not None:
            assert isinstance(t,float), 'epoch time please'
            s['t'].append(t)

        if len(s['y']) > max(2 * k + 5, NPROPHET_META['n_warm']):
            # Offset y, t, a are supplied to nprophet interface
            t_arg = s['t'][k:] if t is not None else None
            a_arg = s['a']
            y_arg = s['y'][k:]
            x, x_std, forecast, model = nprophet_iskater_factory(y=y_arg, k=k, a=a_arg, t=t_arg,
                                                                freq=freq, n_max=n_max,
                                                                recursive=recursive, model_params=model_params)
            s['m'] = True # Flag indicating a model has been fit (there is no point keeping the model itself, however)
        else:
            x = [y[0]] * k
            x_std = None

        # Get running mean prediction errors from the prediction parade
        x_resid, x_resid_std, s['p'] = parade(p=s['p'], x=x, y=y[0])
        x_resid = nonecast(x_resid,y[0])
        x_resid_std = nonecast(x_resid_std,1.0)

        # Compute center of mass between bias-corrected and uncorrected predictions
        x_corrected = np.array(x_resid) + np.array(x)
        x_center = nonecenter(m=[emp_mass, 1 - emp_mass], x=[x_corrected, x])
        x_std_center = nonecenter(m=[emp_std_mass, 1 - emp_std_mass], x=[x_resid_std, x_std])

        return x_center, x_std_center, s


def fbnprophet_hyperparam_skater_factory(r: R_TYPE = None, param_names: [str] = None, **kwargs):
    """ Useful for creating skaters based on hyper-parameters r and the
        method of modifying them suggested by the authors
     """
    assert param_names is not None
    dim = len(param_names)
    #assert 2 <= dim <= 3
    model_params = nprophet_params(r=r,dim=dim, param_names=param_names)
    return fbnprophet_skater_factory(model_params=model_params, **kwargs)


def fbnprophet_skater_testor(y :Y_TYPE, s:dict=None, k:int =1, a:A_TYPE =None,
                     t:T_TYPE=None, e:E_TYPE =None, r:R_TYPE =None, freq=None, n_max=None):
    """ A default facebook nprophet usage, with no hyper-parameters and no prediction parade """
    # For testing

    if freq is None:
        freq = NPROPHET_META['freq']
    if n_max is None:
        n_max = NPROPHET_META['n_max']

    y = wrap(y)
    a = wrap(a)

    if not s.get('y'):
        s = {'y': list(),
             'a': list(),
             'k': k}
    else:
        # Assert immutability of k, dimensions
        if s['y']:
            assert len(y) == len(s['y'][0])
            assert k == s['k']
        if s['a']:
            assert len(a) == len(s['a'][0])

    if y is None:
        return None, s, None
    else:
        s['y'].append(y)
        if a is not None:
            s['a'].append(a)
        if len(s['y']) > max(2*k+5, NPROPHET_META['n_warm']):
            x, x_std, _, _ = nprophet_iskater_factory(y=s['y'], k=k, a=s['a'], freq=freq, n_max=n_max)
        else:
            x = [y[0]] * k
            x_std = [1.0] * k
        return x, x_std, s


if __name__ == '__main__':
    from timemachines.skatertools.data.real import hospital_with_exog
    from timemachines.skatertools.evaluation.evaluators import (
        evaluate_mean_absolute_error, evaluate_mean_squared_error
    )
    import argparse
    
    
    parser = argparse.ArgumentParser(description='Compute error on skater.')
    parser.add_argument(
        '-k', type=int, help='k'
    )
    parser.add_argument(
        '-n', type=int, default=100, help='n'
    )
    parser.add_argument(
        '-offset', type=bool, default=True, help='offset'
    )
    parser.add_argument(
        '-n_burn', type=int, default=50, help='n_burn'
    )
    args = parser.parse_args()
    
    y, a = hospital_with_exog(k=args.k, n=args.n, offset=args.offset)
    f = fbnprophet_skater_factory
    err_abs = evaluate_mean_absolute_error(
        f=f, k=args.k, y=y, a=a, n_burn=args.n_burn
    )
    err_sq = evaluate_mean_squared_error(
        f=f, k=args.k, y=y, a=a, n_burn=args.n_burn
    )
    print('abs error:', err_abs)
    print('sq error:', err_sq)
