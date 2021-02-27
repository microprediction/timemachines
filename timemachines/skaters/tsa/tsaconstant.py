from timemachines.skatertools.visualization.priorplot import prior_plot_exogenous
from statsmodels.tsa.arima.model import ARIMA
from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE, wrap
from typing import Any
from timemachines.skaters.tsa.tsaparams import TSA_META
from timemachines.skatertools.ensembling.ensemblefactory import precision_weighted_ensemble_factory

TSA_P_DEFAULT = 3
TSA_D_DEFAULT = 0
TSA_Q_DEFAULT = 3


def tsa_constant_skater_factory(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                             t: T_TYPE = None, e: E_TYPE = None,
                             p:int=TSA_P_DEFAULT, d:int=TSA_D_DEFAULT, q:int=TSA_D_DEFAULT) -> ([float], Any, Any):
    """ Extremely simple univariate, fixed p,d,q ARIMA model that is re-fit each time """

    # TODO: FIX THIS TO USE EMPIRICAL STD, OTHERWISE ENSEMBLES ARE DREADFUL

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
        if len(s['y']) > max(2 * k + 5, TSA_META['n_warm']):
            y0s = [ y_[0] for y_ in s['y']]
            model = ARIMA(y0s, order=(p,d,q))
            try:
                x = list( model.fit().forecast(steps=k) )
            except:
                x = [wrap(y)[0]]*k
            x_std = [1.0]*k
        else:
            x = [y[0]] * k
            x_std = [1.0] * k
        return x, x_std, s


# TODO: Replace tsa with empirical tsa, otherwise these ensembles are silly


def tsa_constant_skater_p1_d0_q0(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                             t: T_TYPE = None, e: E_TYPE = None) -> ([float], Any, Any):
    return tsa_constant_skater_factory(y=y,s=s,k=k,a=a,t=t,e=e,p=1,d=0,q=0)


def tsa_constant_skater_p2_d0_q0(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                             t: T_TYPE = None, e: E_TYPE = None) -> ([float], Any, Any):
    return tsa_constant_skater_factory(y=y,s=s,k=k,a=a,t=t,e=e,p=2,d=0,q=0)


def tsa_constant_skater_p3_d0_q0(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                             t: T_TYPE = None, e: E_TYPE = None) -> ([float], Any, Any):
    return tsa_constant_skater_factory(y=y,s=s,k=k,a=a,t=t,e=e,p=3,d=0,q=0)

TSA_D0_Q0_SKATERS = [ tsa_constant_skater_p1_d0_q0, tsa_constant_skater_p2_d0_q0,
                      tsa_constant_skater_p3_d0_q0 ]

def tsa_constant_skater_p1_d0_q1(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                             t: T_TYPE = None, e: E_TYPE = None) -> ([float], Any, Any):
    return tsa_constant_skater_factory(y=y,s=s,k=k,a=a,t=t,e=e,p=1,d=0,q=1)


def tsa_constant_skater_p2_d0_q1(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                             t: T_TYPE = None, e: E_TYPE = None) -> ([float], Any, Any):
    return tsa_constant_skater_factory(y=y,s=s,k=k,a=a,t=t,e=e,p=2,d=0,q=1)


def tsa_constant_skater_p3_d0_q1(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                             t: T_TYPE = None, e: E_TYPE = None) -> ([float], Any, Any):
    return tsa_constant_skater_factory(y=y,s=s,k=k,a=a,t=t,e=e,p=3,d=0,q=1)

TSA_D0_Q1_SKATERS = [ tsa_constant_skater_p1_d0_q0, tsa_constant_skater_p2_d0_q0,
                      tsa_constant_skater_p3_d0_q0 ]

TSA_D0_SKATERS = TSA_D0_Q0_SKATERS + TSA_D0_Q1_SKATERS

#################################### d=1 ####################################



def tsa_constant_skater_p1_d1_q0(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                             t: T_TYPE = None, e: E_TYPE = None) -> ([float], Any, Any):
    return tsa_constant_skater_factory(y=y,s=s,k=k,a=a,t=t,e=e,p=1,d=1,q=0)


def tsa_constant_skater_p2_d1_q0(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                             t: T_TYPE = None, e: E_TYPE = None) -> ([float], Any, Any):
    return tsa_constant_skater_factory(y=y,s=s,k=k,a=a,t=t,e=e,p=2,d=1,q=0)


def tsa_constant_skater_p3_d1_q0(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                             t: T_TYPE = None, e: E_TYPE = None) -> ([float], Any, Any):
    return tsa_constant_skater_factory(y=y,s=s,k=k,a=a,t=t,e=e,p=3,d=1,q=0)


TSA_D1_Q0_SKATERS = [ tsa_constant_skater_p1_d1_q0, tsa_constant_skater_p2_d1_q0,
                      tsa_constant_skater_p3_d1_q0 ]

def tsa_constant_skater_p1_d1_q1(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                             t: T_TYPE = None, e: E_TYPE = None) -> ([float], Any, Any):
    return tsa_constant_skater_factory(y=y,s=s,k=k,a=a,t=t,e=e,p=1,d=1,q=1)


def tsa_constant_skater_p2_d1_q1(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                             t: T_TYPE = None, e: E_TYPE = None) -> ([float], Any, Any):
    return tsa_constant_skater_factory(y=y,s=s,k=k,a=a,t=t,e=e,p=2,d=1,q=1)


def tsa_constant_skater_p3_d1_q1(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                             t: T_TYPE = None, e: E_TYPE = None) -> ([float], Any, Any):
    return tsa_constant_skater_factory(y=y,s=s,k=k,a=a,t=t,e=e,p=3,d=1,q=1)


TSA_D1_Q1_SKATERS = [ tsa_constant_skater_p1_d1_q0, tsa_constant_skater_p2_d1_q0,
                      tsa_constant_skater_p3_d1_q0 ]

TSA_D1_SKATERS = TSA_D1_Q0_SKATERS + TSA_D1_Q1_SKATERS

TSA_CONSTANT_SKATERS = TSA_D0_SKATERS + TSA_D1_SKATERS





def tsa_balanced_ensemble(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
    """
         "less than precision" weighted moving averages
    """
    fs = TSA_CONSTANT_SKATERS
    return precision_weighted_ensemble_factory(fs=fs,y=y,s=s,k=k,a=a,t=t,e=e,r=0.25)


def tsa_precise_ensemble(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
    """
         Precision weight moving averages
    """
    fs = TSA_CONSTANT_SKATERS
    return precision_weighted_ensemble_factory(fs=fs,y=y,s=s,k=k,a=a,t=t,e=e,r=0.5)


def tsa_aggressive_ensemble(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
    """
         Precision weight moving averages
    """
    fs = TSA_CONSTANT_SKATERS
    return precision_weighted_ensemble_factory(fs=fs,y=y,s=s,k=k,a=a,t=t,e=e,r=0.75)


def tsa_ma_precise_ensemble(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
    """
         Precision weight moving averages
    """
    from timemachines.skaters.simple.movingaverage import EMA_SKATERS
    fs = TSA_CONSTANT_SKATERS
    return precision_weighted_ensemble_factory(fs=fs,y=y,s=s,k=k,a=a,t=t,e=e,r=0.5)


TSA_ENSEMBLE_SKATERS = [ tsa_aggressive_ensemble, tsa_balanced_ensemble, tsa_precise_ensemble ]

TSA_SKATERS = TSA_ENSEMBLE_SKATERS + TSA_CONSTANT_SKATERS


if __name__=='__main__':
    prior_plot_exogenous(f=tsa_aggressive_ensemble,k=5,n=TSA_META['n_warm']+25,n_plot=50)