from timemachines.skaters.rvr.rvrinclusion import using_river

if using_river:
    from timemachines.skatertools.ensembling.ensemblefactory import precision_weighted_ensemble_factory, R_BALANCED, R_AGGRESSIVE, R_PRECISION
    from timemachines.skaters.rvr.rvrconstant import RIVER_CONSTANT_SKATERS
    from timemachines.skatertools.utilities.conventions import Y_TYPE,A_TYPE,T_TYPE,E_TYPE
    from typing import Any


    def rvr_precision_ensemble(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
        return precision_weighted_ensemble_factory(fs=RIVER_CONSTANT_SKATERS,y=y,s=s,k=k,a=a,t=t,e=e,r=R_PRECISION)


    def rvr_balanced_ensemble(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
        return precision_weighted_ensemble_factory(fs=RIVER_CONSTANT_SKATERS,y=y,s=s,k=k,a=a,t=t,e=e,r=R_BALANCED)


    def rvr_aggressive_ensemble(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
        return precision_weighted_ensemble_factory(fs=RIVER_CONSTANT_SKATERS,y=y,s=s,k=k,a=a,t=t,e=e,r=R_AGGRESSIVE)

    RIVER_ENSEMBLE_SKATERS = [ rvr_aggressive_ensemble, rvr_balanced_ensemble, rvr_aggressive_ensemble ]
else:
    RIVER_ENSEMBLE_SKATERS = []

