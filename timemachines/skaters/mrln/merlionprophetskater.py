
from timemachines.skaters.mrln.merlioninclusion import using_merlion

if using_merlion and False:
    from timemachines.skaters.mrln.merlionwrappers import merlion_Prophet_iskater
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE
    from timemachines.skatertools.batch.batchskater import batch_skater_factory

    def merlion_prophet(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None, emp_mass=0.0):
        return batch_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, emp_mass=emp_mass,
                                    iskater=merlion_Prophet_iskater,
                                    iskater_kwargs={},
                                    min_e=0)

    MERLION_PROPHET_SKATERS = [merlion_prophet]

else:
    MERLION_PROPHET_SKATERS = []

if __name__=='__main__':
    from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error_with_sporadic_fit
    err1 = hospital_mean_square_error_with_sporadic_fit(f=merlion_Prophet_iskater, k=3, n=110)