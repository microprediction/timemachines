from timemachines.skaters.rvr.rvrinclusion import using_river
from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE, wrap
from typing import Any
from timemachines.skatertools.visualization.priorplot import prior_plot_exogenous

if using_river:
    from timemachines.skaters.rvr.rvrsarimax import rvr_sarimax_factory

    def rvr_p1_d0_q0(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                             t: T_TYPE = None, e: E_TYPE = None) -> ([float], Any, Any):
        return rvr_sarimax_factory(y=y,s=s,k=k,a=a,t=t,e=e,p=1,d=0,q=0)


    def rvr_p2_d0_q0(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                     t: T_TYPE = None, e: E_TYPE = None) -> ([float], Any, Any):
        return rvr_sarimax_factory(y=y, s=s, k=k, a=a, t=t, e=e, p=2, d=0, q=0)


    def rvr_p3_d0_q0(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                     t: T_TYPE = None, e: E_TYPE = None) -> ([float], Any, Any):
        return rvr_sarimax_factory(y=y, s=s, k=k, a=a, t=t, e=e, p=3, d=0, q=0)


    def rvr_p5_d0_q0(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                     t: T_TYPE = None, e: E_TYPE = None) -> ([float], Any, Any):
        return rvr_sarimax_factory(y=y, s=s, k=k, a=a, t=t, e=e, p=5, d=0, q=0)

    def rvr_p8_d0_q0(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                     t: T_TYPE = None, e: E_TYPE = None) -> ([float], Any, Any):
        return rvr_sarimax_factory(y=y, s=s, k=k, a=a, t=t, e=e, p=5, d=0, q=0)


    RIVER_CONSTANT_SKATERS = [rvr_p1_d0_q0, rvr_p2_d0_q0, rvr_p3_d0_q0, rvr_p3_d0_q0, rvr_p5_d0_q0, rvr_p8_d0_q0]
else:
    RIVER_CONSTANT_SKATERS = []

if __name__=='__main__':
    prior_plot_exogenous(f=rvr_p1_d0_q0,k=5,n=50,n_plot=50)