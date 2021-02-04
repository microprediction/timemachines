from timemachines.skaters.dlm.dlmunivariate import dlm_univariate_a, dlm_univariate_r3
from timemachines.skaters.evaluation import hospital_mean_square_error, hospital_exog_mean_square_error


def test_dlm_auto_univariate():
    print(hospital_mean_square_error(f=dlm_univariate_a, k=3, n=150, n_burn=44))
    print(hospital_mean_square_error(f=dlm_univariate_r3, k=3, n=150, n_burn=44, r=0.53))


if __name__=='__main__':
    test_dlm_auto_univariate()