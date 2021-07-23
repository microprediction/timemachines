from timemachines.skatertools.visualization.priorplot import prior_plot_exogenous
from statsmodels.tsa.forecasting.theta import ThetaModel
from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE, wrap
from typing import Any
from timemachines.skaters.tsa.tsaparams import TSA_META
from timemachines.skatertools.ensembling.ensemblefactory import precision_weighted_ensemble_factory
from timemachines.skatertools.components.parade import parade
from timemachines.skatertools.utilities.nonemath import nonecast
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)


def tsa_theta_factory(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                             t: T_TYPE = None, e: E_TYPE = None,
                      method=None, deseasonalize=True,
                      use_test=True, difference=False) -> ([float], Any, Any):
    """

    period : int, default None
        The period of the data that is used in the seasonality test and
        adjustment. If None then the period is determined from y's index,
        if available.
    deseasonalize : bool, default True
        A flag indicating whether the deseasonalize the data. If True and
        use_test is True, the data is only deseasonalized if the null of no
        seasonal component is rejected.
    use_test : bool, default True
        A flag indicating whether test the period-th autocorrelation. If this
        test rejects using a size of 10%, then decomposition is used. Set to
        False to skip the test.
    method : {"auto", "additive", "multiplicative"}, default "auto"
        The model used for the seasonal decomposition. "auto" uses a
        multiplicative if y is non-negative and all estimated seasonal
        components are positive. If either of these conditions is False,
        then it uses an additive decomposition.
    difference : bool, default False
        A flag indicating to difference the data before offlinetesting for
        seasonality."""

    y = wrap(y)
    a = wrap(a)

    if not s.get('y'):
        s = {'y': list(),
             'a': list(),
             'k': k,
             'p':{}}
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
        if len(s['y']) > max(2 * k + 5, TSA_META['n_warm']) and (e is not None and e>0):
            y0s = [ y_[0] for y_ in s['y']]
            import numpy as np
            model = ThetaModel(np.array(y0s), method=method, deseasonalize=deseasonalize,
                      use_test=use_test, difference=difference)
            try:
                x = list( model.fit().forecast(steps=k) )
            except Exception as e:
                print(e)
                x = [wrap(y)[0]]*k
        else:
            x = [y[0]] * k

        y0 = wrap(y)[0]
        _we_ignore_bias, x_std, s['p'] = parade(p=s['p'], x=x, y=y0)
        x_std_fallback = nonecast(x_std, fill_value=1.0)
        return x, x_std_fallback, s


def tsa_theta_auto(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                             t: T_TYPE = None, e: E_TYPE = None) -> ([float], Any, Any):
    return tsa_theta_factory(y=y,s=s,k=k,a=a,t=t,e=e, method='auto', deseasonalize=False)


def tsa_theta_additive(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                             t: T_TYPE = None, e: E_TYPE = None) -> ([float], Any, Any):
    return tsa_theta_factory(y=y,s=s,k=k,a=a,t=t,e=e, method='additive', deseasonalize=False)


def tsa_theta_multiplicative(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                           t: T_TYPE = None, e: E_TYPE = None) -> ([float], Any, Any):
        return tsa_theta_factory(y=y, s=s, k=k, a=a, t=t, e=e, method='multiplicative', deseasonalize=False)


TSA_THETA_SKATERS = [ tsa_theta_auto, tsa_theta_additive, tsa_theta_multiplicative ]



if __name__=='__main__':
    prior_plot_exogenous(f=tsa_theta_auto,k=1,n=TSA_META['n_warm']+25,n_plot=25)
