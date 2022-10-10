from timemachines.skaters.sk.skinclusion import using_sktime
from timemachines.skaters.sk.sfinclusion import using_statsforecast

# See https://github.com/sktime/sktime/blob/main/sktime/forecasting/statsforecast.py


if using_sktime and using_statsforecast:
    from timemachines.skaters.sk.skwrappers import sf_autoarima_iskater
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE
    from timemachines.skatertools.batch.batchskater import batch_skater_factory

    def sf_autoarima(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None, emp_mass=0.0):
        return batch_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, emp_mass=emp_mass,
                                    iskater=sf_autoarima_iskater,
                                    iskater_kwargs={},
                                    min_e=0)

    SF_AA_SKATERS = [sf_autoarima]

else:
    SF_AA_SKATERS = []




if __name__=='__main__':
    from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error_with_sporadic_fit
    err1 = hospital_mean_square_error_with_sporadic_fit(f=sf_autoarima, k=3, n=110)
