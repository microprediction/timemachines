from timemachines.skaters.nproph.nprophetinclusion import using_neuralprophet,NeuralProphet
if using_neuralprophet:
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE, wrap
    from typing import Any
    from timemachines.skatertools.components.parade import parade
    from timemachines.skaters.nproph.nprophetiskaterfactory import using_neuralprophet
    from timemachines.skaters.nproph.nprophetiskaterfactory import nprophet_iskater_factory
    from timemachines.skatertools.utilities.nonemath import nonecenter
    from timemachines.skaters.nproph.nprophparams import NPROPHET_META
    from timemachines.skatertools.utilities.nonemath import nonecast
    import sys
    import logging
    import numpy as np

    logging.disable(sys.maxsize)
    logging.getLogger('fbprophet').setLevel(logging.ERROR)


    def nprophet_skater_factory(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                                 t: T_TYPE = None, e: E_TYPE = None,
                                 emp_mass: float = 0.0, emp_std_mass: float = 1.0,
                                 freq=None, recursive: bool = False,
                                 model_params: dict = None,    # n_lags ...
                                 n_max: int = None) -> ([float], Any, Any):
        """ Neural prophet skater with running prediction error moments
            Hyper-parameters are explicit here, whereas they are determined from r in actual skaters.
            Params of note:

                 a: value of known-in-advance vars k step in advance (not contemporaneous with y)
                 emp_mass: How much to auto-correct bias
                 emp_std_mass: How much to use empirical std versus model

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

            if len(s['y']) > max(2 * k + 5, NPROPHET_META['n_warm']) and (e is not None and e>0):
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


if __name__=='__main__':
    from timemachines.skatertools.data.real import hospital
    from timemachines.skatertools.evaluation.evaluators import evaluate_mean_absolute_error

    k = 5
    y = hospital(n=420)
    f = nprophet_skater_factory
    err2 = evaluate_mean_absolute_error(f=f, k=k, y=y, n_burn=400)
    print(err2)