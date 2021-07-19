from timemachines.skaters.dlm.dlmunivariate import using_dlm
if using_dlm:

    from timemachines.skaters.dlm.dlmunivariate import dlm_univariate_a, dlm_univariate_r3
    from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error_with_sporadic_fit


    def test_dlm_auto_univariate():
        print(hospital_mean_square_error_with_sporadic_fit(f=dlm_univariate_a, k=3, n=150, fit_frequency=100, n_test=1))
        print(hospital_mean_square_error_with_sporadic_fit(f=dlm_univariate_r3, k=3, n=150, fit_frequency=100, n_test=1, r=0.53))


if __name__=='__main__':
    test_dlm_auto_univariate()