from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE
from timemachines.skaters.bats.batsinclusion import using_bats
if using_bats:
    from timemachines.skaters.bats.batsfactory import bats_factory


    def bats_fast(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None):
        return bats_factory(y=y, s=s, k=k, a=a, t=t, e=e, use_trend=False, use_arma_errors=False, use_box_cox=False)


    def bats_trendy(y :Y_TYPE, s, k:int, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None, r:R_TYPE=None):
        return bats_factory(y=y, s=s, k=k, a=a, t=t, e=e, use_trend=True, use_arma_errors=False, use_box_cox=False)


    def bats_damped(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None):
        return bats_factory(y=y, s=s, k=k, a=a, t=t, e=e, use_damped_trend=True, use_arma_errors=False, use_box_cox=False)


    def bats_arma(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None):
        return bats_factory(y=y, s=s, k=k, a=a, t=t, e=e, use_trend=False, use_arma_errors=True, use_box_cox=False)


    def bats_arma_bc(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None):
        return bats_factory(y=y, s=s, k=k, a=a, t=t, e=e, use_trend=False, use_arma_errors=True, use_box_cox=True)


    def bats_damped_arma(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None):
        return bats_factory(y=y, s=s, k=k, a=a, t=t, e=e, use_damped_trend=True, use_arma_errors=True, use_box_cox=False)


    def bats_trendy_arma(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None):
        return bats_factory(y=y, s=s, k=k, a=a, t=t, e=e, use_trend=True, use_arma_errors=True, use_box_cox=False)


    def bats_bc(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None):
        return bats_factory(y=y, s=s, k=k, a=a, t=t, e=e, use_trend=False, use_arma_errors=False, use_box_cox=True)


    def bats_trendy_bc(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None):
        return bats_factory(y=y, s=s, k=k, a=a, t=t, e=e, use_trend=True, use_arma_errors=False,
                            use_box_cox=True)

    def bats_damped_bc(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None):
        return bats_factory(y=y, s=s, k=k, a=a, t=t, e=e, use_damped_trend=True, use_arma_errors=False,
                            use_box_cox=True)

    def bats_damped_arma_bc(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None):
        return bats_factory(y=y, s=s, k=k, a=a, t=t, e=e, use_damped_trend=True, use_arma_errors=True, use_box_cox=True)


    def bats_trendy_arma_bc(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None):
        return bats_factory(y=y, s=s, k=k, a=a, t=t, e=e, use_trend=True, use_arma_errors=True, use_box_cox=True)


    BATS_SKATERS = [bats_fast, bats_arma, bats_bc, bats_arma_bc,
                    bats_trendy, bats_trendy_arma, bats_trendy_bc, bats_trendy_arma_bc,
                    bats_damped, bats_damped_arma, bats_damped_bc, bats_damped_arma_bc]
else:
    BATS_SKATERS = []