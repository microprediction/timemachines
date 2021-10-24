from timemachines.skaters.pycrt.pycaretinclusion import using_pycaret


if using_pycaret:
    from timemachines.skaters.pycrt.pycaretwrapper import pycrt_iskater
    from timemachines.skatertools.batch.batchskater import batch_skater_factory
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE


    def pycrt_median_3(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None,
                       emp_mass=0.0):
        return batch_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, emp_mass=emp_mass,
                                    iskater=pycrt_iskater,
                                    iskater_kwargs={'n_select': 3,'blend_method':'median'},
                                    min_e=0)


    def pycrt_median_3_full(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None,
                       emp_mass=0.0):
        return batch_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, emp_mass=emp_mass,
                                    iskater=pycrt_iskater,
                                    iskater_kwargs={'n_select': 3, 'blend_method': 'median','turbo':False},
                                    min_e=0)


    def pycrt_median_8(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None, emp_mass=0.0):
        return batch_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, emp_mass=emp_mass,
                                    iskater=pycrt_iskater,
                                    iskater_kwargs={'n_select':8,'blend_method':'median'},
                                    min_e=0)

    def pycrt_mean_3(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None, emp_mass=0.0):
        return batch_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, emp_mass=emp_mass,
                                    iskater=pycrt_iskater,
                                    iskater_kwargs={'n_select':3,'blend_method':'mean'},
                                    min_e=0)

    def pycrt_mean_8(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None, emp_mass=0.0):
        return batch_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, emp_mass=emp_mass,
                                    iskater=pycrt_iskater,
                                    iskater_kwargs={'n_select':8,'blend_method':'mean'},
                                    min_e=0)

    PYCRT_MEDIAN_SKATERS = [pycrt_median_3, pycrt_median_8, pycrt_mean_3, pycrt_mean_8, pycrt_median_3_full]

else:
    PYCRT_MEDIAN_SKATERS = []

if __name__=='__main__':
    from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error_with_sporadic_fit
    err1 = hospital_mean_square_error_with_sporadic_fit(f=pycrt_median_3, k=3, n=110)
