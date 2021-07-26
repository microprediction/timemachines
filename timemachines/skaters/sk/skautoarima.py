from timemachines.skaters.sk.skinclusion import using_sktime
from timemachines.skaters.pmd.pmdinclusion import using_pmd

if using_sktime and using_pmd:
    from timemachines.skaters.sk.skwrappers import sk_autoarima_iskater
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE
    from timemachines.skatertools.batch.batchskater import batch_skater_factory

    def sk_autoarima(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None, emp_mass=0.0):
        return batch_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, emp_mass=emp_mass,
                                    iskater=sk_autoarima_iskater,
                                    iskater_kwargs={},
                                    min_e=0)

    SK_AA_SKATERS = [sk_autoarima]

else:
    SK_AA_SKATERS = []

if __name__=='__main__':
    from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error_with_sporadic_fit
    err1 = hospital_mean_square_error_with_sporadic_fit(f=sk_autoarima, k=3, n=110)
