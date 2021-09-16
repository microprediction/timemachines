
from timemachines.skaters.drts.dartsinclusion import using_darts

if using_darts:
    from timemachines.skaters.drts.dartswrappers import darts_ARIMA_iskater
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE
    from timemachines.skatertools.batch.batchskater import batch_skater_factory

    def darts_arima(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None, emp_mass=0.0):
        return batch_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, emp_mass=emp_mass,
                                    iskater=darts_ARIMA_iskater,
                                    iskater_kwargs={},
                                    min_e=0)

    DARTS_ARIMA_SKATERS = [darts_arima]

else:
    DARTS_ARIMA_SKATERS = []

if __name__=='__main__':
    from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error_with_sporadic_fit
    err1 = hospital_mean_square_error_with_sporadic_fit(f=darts_ARIMA_iskater, k=3, n=110)