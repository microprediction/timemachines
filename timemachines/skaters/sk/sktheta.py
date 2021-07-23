from timemachines.skaters.sk.skinclusion import using_sktime


if using_sktime:
    from timemachines.skaters.sk.skwrappers import sk_theta_hourly_iskater
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE
    from timemachines.skatertools.batch.batchskater import batch_skater_factory

    def sk_theta(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None, emp_mass=0.0,
                 deseasonalize=False):
        return batch_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, emp_mass=emp_mass,
                                    iskater=sk_theta_hourly_iskater,
                                    iskater_kwargs={'deseasonalize':deseasonalize},
                                    min_e=0, n_warm=20)

    SK_THETA_SKATERS = [sk_theta]

else:
    SK_THETA_SKATERS = []

if __name__=='__main__':
    from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error
    err1 = hospital_mean_square_error(f=sk_theta, k=3)
