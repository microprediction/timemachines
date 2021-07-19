from timemachines.skaters.dlm.dlmunivariate import using_dlm
if using_dlm:
    from timemachines.skaters.dlm.dlmexogenous import dlm_exogenous_r3, dlm_exogenous_b
    from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error_with_sporadic_fit, hospital_exog_mean_square_error_with_sporadic_fit
    import numpy as np


    def test_dlm_exogenous():
        print(hospital_exog_mean_square_error_with_sporadic_fit(f=dlm_exogenous_r3, k=1, n=150, fit_frequency=100, n_test=1, r=np.random.rand()))
        print(hospital_exog_mean_square_error_with_sporadic_fit(f=dlm_exogenous_b, k=1, n=150, fit_frequency=100, n_test=1))


    def test_dlm_exogenous_on_univariate():
        print(hospital_mean_square_error_with_sporadic_fit(f=dlm_exogenous_b, k=1, n=150, fit_frequency=100, n_test=1))


if __name__=='__main__':
    test_dlm_exogenous_on_univariate()