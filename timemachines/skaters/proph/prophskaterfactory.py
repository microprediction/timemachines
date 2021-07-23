from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE, wrap
from typing import Any
from timemachines.skatertools.components.parade import parade
from timemachines.skaters.proph.prophiskaterfactory import using_prophet
if using_prophet:
    from timemachines.skaters.proph.prophiskaterfactory import prophet_iskater_factory
from timemachines.skatertools.utilities.nonemath import nonecenter
from timemachines.skaters.proph.prophparams import PROPHET_META, prophet_params
from timemachines.skatertools.utilities.nonemath import nonecast
import sys
import logging
import numpy as np

logging.disable(sys.maxsize)
logging.getLogger('fbprophet').setLevel(logging.ERROR)


###################################################################################################
#                                                                                                 #
#                      Facebook Prophet skater factory                                            #
#                                                                                                 #
# Wraps prophet model with recursive use, running empirical moments, and some hyper-parameters    #
# See https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning              #
#                                                                                                 #
# Advantages:                                                                                     #
#       - State is a simple dictionary, since prophet itself has no notion of state               #
#       - Supposed to work okay without any hyper-param tuning                                    #
#       - Great documentation, response to GitHub issues and backing by Facebook devs             #
#                                                                                                 #
# Disadvantages:                                                                                  #
#       - Model requires re-fitting after each and every data point                               #
#       - Generative model isn't suggestive of strong out of sample performance (opinion)         #
#       - Empirical results are quizzical                                                         #
#                                                                                                 #
# See also: prophetiskaterfatory                                                                  #                              #
#                                                                                                 #
###################################################################################################

if using_prophet:

    def fbprophet_skater_factory(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
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
        if e is None:
          e = 1  # 'Providing e=None to prophet is not recommended. Use e<0 to skip a fitting or e>0 to fit.'

        if freq is None:
            freq = PROPHET_META['freq']
        if n_max is None:
            n_max = PROPHET_META['n_max']

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

            if e is not None and (e>0) and ((a is None) or (len(s['a'])>k)) and (len(s['y'])>10):
                # Offset y, t, a are supplied to prophet interface
                # Prophet requires several non-nan rows
                t_arg = s['t'][k:] if t is not None else None
                a_arg = s['a']
                y_arg = s['y'][k:]
                x, x_std, forecast, model = prophet_iskater_factory(y=y_arg, k=k, a=a_arg, t=t_arg,
                                                                    freq=freq, n_max=n_max,
                                                                    recursive=recursive, model_params=model_params)
                s['m'] = True # Flag indicating a model has been fit (there is no point keeping the model itself, however)
            else:
                x = [y[0]] * k
                x_std = [1] * k

            # Get running mean prediction errors from the prediction parade
            x_resid, x_resid_std, s['p'] = parade(p=s['p'], x=x, y=y[0])
            x_resid = nonecast(x_resid,y[0])
            x_resid_std = nonecast(x_resid_std,1.0)

            # Compute center of mass between bias-corrected and uncorrected predictions
            x_corrected = np.array(x_resid) + np.array(x)
            x_center = nonecenter(m=[emp_mass, 1 - emp_mass], x=[x_corrected, x])
            x_std_center = nonecenter(m=[emp_std_mass, 1 - emp_std_mass], x=[x_resid_std, x_std])

            return x_center, x_std_center, s


    def fbprophet_hyperparam_skater_factory(r: R_TYPE = None, param_names: [str] = None,   **kwargs):
        """ Useful for creating skaters based on hyper-parameters r and the
            method of modifying them suggested by the authors
         """
        assert param_names is not None
        dim = len(param_names)
        assert 2 <= dim <= 3
        model_params = prophet_params(r=r,dim=dim, param_names=param_names)
        return fbprophet_skater_factory(model_params=model_params, **kwargs)


    def fbprophet_skater_testor(y :Y_TYPE, s:dict=None, k:int =1, a:A_TYPE =None,
                         t:T_TYPE=None, e:E_TYPE =None, r:R_TYPE =None, freq=None, n_max=None):
        """ A default facebook prophet usage, with no hyper-parameters and no prediction parade """
        # For offlinetesting

        if freq is None:
            freq = PROPHET_META['freq']
        if n_max is None:
            n_max = PROPHET_META['n_max']

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
            if len(s['y']) > max(2*k+5,PROPHET_META['n_warm']):
                x, x_std, _, _ = prophet_iskater_factory(y=s['y'], k=k, a=s['a'], freq=freq, n_max=n_max)
            else:
                x = [y[0]] * k
                x_std = [1.0] * k
            return x, x_std, s


if __name__ == '__main__':
    from timemachines.skatertools.data import hospital_with_exog
    from timemachines.skatertools.evaluation.evaluators import evaluate_mean_absolute_error

    k = 3
    y, a = hospital_with_exog(k=k, n=100, offset=True)
    f = fbprophet_skater_factory
    err2 = evaluate_mean_absolute_error(f=f, k=k, y=y, a=a, n_burn=50)
    print(err2)
