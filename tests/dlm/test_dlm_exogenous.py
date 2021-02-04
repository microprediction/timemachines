from timemachines.skaters.dlm.dlmexogenous import dlm_exogenous_r3, dlm_exogenous_b
from timemachines.skaters.evaluation import hospital_mean_square_error, hospital_exog_mean_square_error
import numpy as np


def test_dlm_exogenous():
    print(hospital_exog_mean_square_error(f=dlm_exogenous_r3, k=1,n=150, n_burn=44, r=np.random.rand()))
    print(hospital_exog_mean_square_error(f=dlm_exogenous_b, k=1,n=150, n_burn=44))


def test_dlm_exogenous_on_univariate():
    print(hospital_mean_square_error(f=dlm_exogenous_b, k=1,n=150, n_burn=44))


if __name__=='__main__':
    test_dlm_exogenous_on_univariate()