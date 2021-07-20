from timemachines.skatertools.ensembling.ensemblefactory import precision_weighted_ensemble_factory, R_BALANCED, R_AGGRESSIVE, R_PRECISION
from timemachines.skaters.tsa.tsaconstant import TSA_D0_SKATERS
from timemachines.skaters.tsa.tsatheta import TSA_THETA_SKATERS
from timemachines.skatertools.utilities.conventions import Y_TYPE,A_TYPE,T_TYPE,E_TYPE
from typing import Any


def tsa_precision_d0_ensemble(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
    return precision_weighted_ensemble_factory(fs=TSA_D0_SKATERS,y=y,s=s,k=k,a=a,t=t,e=e,r=R_PRECISION)


def tsa_balanced_d0_ensemble(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
    return precision_weighted_ensemble_factory(fs=TSA_D0_SKATERS,y=y,s=s,k=k,a=a,t=t,e=e,r=R_BALANCED)


def tsa_aggressive_d0_ensemble(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
    return precision_weighted_ensemble_factory(fs=TSA_D0_SKATERS,y=y,s=s,k=k,a=a,t=t,e=e,r=R_AGGRESSIVE)


def tsa_precision_theta_ensemble(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
    return precision_weighted_ensemble_factory(fs=TSA_THETA_SKATERS,y=y,s=s,k=k,a=a,t=t,e=e,r=R_PRECISION)


def tsa_balanced_theta_ensemble(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
    return precision_weighted_ensemble_factory(fs=TSA_THETA_SKATERS,y=y,s=s,k=k,a=a,t=t,e=e,r=R_BALANCED)


def tsa_aggressive_theta_ensemble(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
    return precision_weighted_ensemble_factory(fs=TSA_THETA_SKATERS,y=y,s=s,k=k,a=a,t=t,e=e,r=R_AGGRESSIVE)


def tsa_precision_combined_ensemble(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
    return precision_weighted_ensemble_factory(fs=TSA_THETA_SKATERS+TSA_D0_SKATERS,y=y,s=s,k=k,a=a,t=t,e=e,r=R_PRECISION)


def tsa_balanced_combined_ensemble(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
    return precision_weighted_ensemble_factory(fs=TSA_THETA_SKATERS+TSA_D0_SKATERS,y=y,s=s,k=k,a=a,t=t,e=e,r=R_BALANCED)


def tsa_aggressive_combined_ensemble(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
    return precision_weighted_ensemble_factory(fs=TSA_THETA_SKATERS+TSA_D0_SKATERS,y=y,s=s,k=k,a=a,t=t,e=e,r=R_AGGRESSIVE)


TSA_DO_ENSEMBLE_SKATERS  = [tsa_precision_d0_ensemble, tsa_balanced_d0_ensemble, tsa_aggressive_d0_ensemble]

TSA_THETA_ENSEMBLE_SKATERS = [ tsa_precision_theta_ensemble, tsa_balanced_theta_ensemble, tsa_aggressive_theta_ensemble ]

TSA_COMBINED_ENSEMBLE_SKATERS = [tsa_precision_combined_ensemble, tsa_balanced_combined_ensemble, tsa_aggressive_combined_ensemble ]

TSA_ENSEMBLE_SKATERS = TSA_DO_ENSEMBLE_SKATERS + TSA_THETA_ENSEMBLE_SKATERS + TSA_COMBINED_ENSEMBLE_SKATERS
